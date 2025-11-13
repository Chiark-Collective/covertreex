I feel your pain. I read through the docs you pasted (thanks for the detail) and I’ll give you a straight, un‑biased read: there *are* a few fundamental, high‑leverage issues that can easily make a 32k build take minutes—even if the “algorithm” is sound. Below I’ll point to specific patterns in your write‑up and call out where the time is likely being burned, then propose surgical fixes (some are 1–3 line patches) and a short order of experiments to confirm.

---

## Where the minutes are really going (from your own telemetry)

### 1) Residual builds are dominated by traversal + a Pythonish kernel path

* Dense residual (gate off) shows **≈257 s** total with **~1.2–1.4 s per dominated batch traversal** *and* **~2.5 s per batch conflict graph** in the Nov 10 snapshot.
* Sparse residual + scope cap shows **~493 s** with **~8.65 s traversal per dominated batch**, while conflict graph drops to **~0.038 s**.
  → This contrast screams that when you *don’t* reuse a cached pairwise tile and instead go back to a kernel slice during adjacency, you pay **seconds per batch**. (You even note this earlier: “the residual adjacency filter still recomputes pairwise kernels…”).

### 2) Your “kernel provider” sits on the hot path

You explicitly route traversal through: *request raw kernel block → feed chunk kernel*. If `kernel_provider(rows, cols)` is a Python callable that:

* slices `X[rows]` / `X[cols]` each time,
* computes RBF with **3‑D broadcasting** (common anti‑pattern),
* returns **float64** temporaries,

…you’ll allocate 100–200 MB tiles repeatedly and do all of that under the GIL. That alone can account for *many* seconds per dominated batch.

> Cost intuition: a 512×8192 tile at d=8 via broadcasting builds a (512,8192,8) temporary ≈ 33.6 M floats. At float64 that’s ~269 MB per call; at float32 ~134 MB. Even ignoring math, that allocation + GC churn can cough up seconds if repeated in a loop. With GEMM‑style (dot + row‑norm identity) it’s a few tens of milliseconds.

### 3) Pairwise reuse is inconsistent across code paths

Some places you say reuse is “enforced”; elsewhere the own numbers show conflict graph still at **~2.5 s/batch** on dense residual. The sparse run (conflict ~0.038 s) is proof that **when** the residual pairwise block is available and reused, the conflict stage becomes a blip. Dense needs the same treatment.

### 4) Float64/contiguity thrash

You whiten to float32 for the gate, but elsewhere `v_matrix`, `p_diag`, `kernel_diag`, and tiles may still be float64 / non‑contiguous. On 32k, bandwidth dwarfs flops—doubling the bytes often doubles the time.

### 5) Sorting where a selection would do

Your “semisort” counters (0.3 s median / batch in dense; tens of seconds in old sparse) likely still perform a full `argsort`. For k‑NN / small‑k tasks, `np.argpartition` + a tiny local sort of the top‑L beats full sorts by an order of magnitude.

---

## Concrete fixes (high leverage first)

### ✅ Fix 1 — Kill Python and broadcasting inside `kernel_provider`

**Goal:** make every kernel tile a pure BLAS + vectorized path, zero Python loops, zero 3‑D broadcasting, and **float32** all the way.

**What to implement:**

* Preconvert inputs once: `X = np.asarray(X, dtype=np.float32, order='C'); row_norm = (X**2).sum(1)` and stash them in `ResidualCorrHostData` (or a sibling).
* Build the tile via the identity `‖x−y‖² = ‖x‖² + ‖y‖² − 2·x·yᵀ` using GEMM:

```python
# Vectorized RBF provider with GEMM (no 3-D broadcast, no Python loops)
def make_rbf_provider(X_f32: np.ndarray, gamma: float):
    X = X_f32  # (N, d), C-contiguous, float32
    r = (X * X).sum(axis=1)  # (N,), float32

    def provider(rows: np.ndarray, cols: np.ndarray, out=None):
        # rows, cols are int64/32 index arrays
        A = X[rows]                  # (m, d)
        B = X[cols]                  # (n, d)
        # Work buffer (optional) to avoid re-allocs:
        ABt = A @ B.T                # (m, n) via SGEMM
        # d2 = ||x||^2 + ||y||^2 - 2 x y^T
        d2 = r[rows][:, None] + r[cols][None, :] - 2.0 * ABt
        if out is None:
            out = np.empty_like(d2, dtype=np.float32)
        # K = exp(-gamma * d2)
        np.exp(-gamma * d2, out=out)  # in-place, float32
        return out
    return provider
```

**Why this matters:** every place that currently does a Python‑level `kernel_provider(rows, cols)` with broadcasting should flip to the above provider. On a modern CPU, a 512×8192×8 GEMM in float32 is **milliseconds**; broadcast‑based RBF can be hundreds of milliseconds to seconds.

> **Checklist:** audit your tree for *any* usage of:
>
> * `X[rows][:, None, :] - X[cols][None, :, :]` (or similar)
> * Python loops around RBF
> * Implicit float64 creation (default dtype)
>
> Replace them with the GEMM provider, and keep a reusable `out` buffer per expected tile size to avoid allocator churn.

---

### ✅ Fix 2 — Thread the cached residual pairwise block into the **dense** conflict graph

You already did this for sparse; bring dense to parity.

**What to change:**

* Make `build_conflict_graph(..., residual_pairwise=None)` a **hard error** if metric is residual and the pairwise block for the batch is missing. Don’t let the adjacency builder compute any new kernel—ever.
* Pass `residual_pairwise[owners, owners]` (or the exact batch id order you use) straight into the adjacency Numba helper.
* Ensure the builder never creates its own tiles from a provider; with the change above, the only place that computes new kernel tiles is traversal, *once*.

**Why this matters:** Your dense‑residual “conflict_graph_ms≈2.51 s” almost certainly comes from recomputing kernels there. The sparse path proves that when a cached block is present, conflict graph time collapses to ~40 ms.

---

### ✅ Fix 3 — Normalize dtypes & layout at configuration time

Make the whole residual path **float32, C‑contiguous** unless correctness demands otherwise.

One place to do it safely:

```python
def configure_residual_correlation(...):
    v = np.asarray(v_matrix, dtype=np.float32, order='C')
    p = np.asarray(p_diag,   dtype=np.float32, order='C')
    kd = np.asarray(kernel_diag, dtype=np.float32, order='C')
    # whitened gate caches already float32 — verify once, not per batch
    return ResidualCorrHostData(v_matrix=v, p_diag=p, kernel_diag=kd, ...)
```

Also verify that any `np.take`/fancy indexing that produces non‑contiguous views is immediately copied to contiguous before passing to Numba kernels (to avoid silent temporary copies *inside* the kernel call).

---

### ✅ Fix 4 — Replace full sorts with selection

Anywhere you pick “top‑k” or split by a small threshold:

* Use `np.argpartition(arr, k)` to get the k smallest in O(n).
* Then `np.argsort` on just those k to order them.

We’ve routinely seen **5–20×** speedups here. Your `traversal_semisort_ms` will flatline.

---

### ✅ Fix 5 — Remove Python callouts from Numba paths (decoder, small helpers)

* `point_decoder` must not be a Python callable touched in tight loops. If you need a mapping, pass an `np.ndarray[int32]` of dataset ids and index it in Numba.
* Audit `compute_residual_distances_with_radius`: ensure it’s `@njit(nopython=True, fastmath=True)` and that all inputs are plain ndarrays, not objects.

---

### ✅ Fix 6 — Threading / oversubscription sanity

If you use Numba *and* MKL/OpenBLAS at the same time, oversubscription can wreck throughput.

Try (for CPU builds):

```bash
export OMP_NUM_THREADS=<# physical cores>
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue  # or tbb if you ship it
```

If your chunk kernel uses `parallel=True`, disable MKL threads so you don’t nest BLAS + Numba threads.

---

### ✅ Fix 7 — Chunk size & workspace tuning

* 512 is a fine default, but on many CPUs 1024 (or 768) reduces call overhead and better aligns with L2/L3. Once the provider is GEMM‑based, it’s cheap to sweep `chunk_size` vs. end‑to‑end time.
* Preallocate work buffers (e.g., `ABt`, `d2`, `K`) in a small `Workspace` struct and reuse them across calls to eliminate allocator pressure.

---

## Quick wins you can land *immediately* (likely 2–10× on residual builds)

1. **Swap in the GEMM‑based `kernel_provider` (Fix 1).**
   This alone often turns “seconds per tile” into “milliseconds per tile”. It touches traversal and any place that still asks for kernel tiles.

2. **Make conflict graph refuse to run without the cached batch pairwise (Fix 2).**
   Dense residual will stop paying that ~2.5 s/batch tax.

3. **Float32 + contiguous everywhere (Fix 3).**
   Halves memory traffic, avoids hidden copies. Easy to enforce in one place.

If you do just those three, the residual build should drop from **minutes** to **tens of seconds** on the 32k harness—*without* relying on gates, prefilters, or algorithmic changes.

---

## “Are we accidentally building on the wrong set?”

One more subtle trap that can turn O(b²) into O(S²): make absolutely sure the **conflict graph** is built over the **batch of new nodes** (size ≈512), not the union of their *scopes*. Your write‑up mentions “owner/member pairs” and CSR streaming—great for scopes—but the conflict graph among *owners* should be tiny:

* if `n_batch=512`, pairwise is ~262k entries; trivial to compute.
* If you accidentally form adjacency over big scope unions (millions), you’ll square that number and explode time. Double‑check the argument you feed into the builder.

---

## Instrumentation to prove it (five lines)

Add simple timers/counters so you can “see” the hotspots in one run:

```python
# around every kernel_provider call
t0 = perf_counter()
K_tile = kernel_provider(rows, cols, out=workspace.K)
telemetry.kernel_provider_ms += (perf_counter() - t0) * 1e3
telemetry.kernel_tiles += 1
telemetry.kernel_elems += K_tile.size
```

and

```python
# in build_conflict_graph dense path
assert residual_pairwise is not None, "Residual: conflict graph requires cached pairwise"
```

One good run will tell you if provider time collapses after Fix 1, and whether conflict still burns time after Fix 2.

---

## Why gates & “scope caps” felt like minutiae

They’re band‑aids on the *amount* of work. The **nature** of the work (Python call overhead, broadcast temporaries, float64 bandwidth, recomputing kernels in conflict) is what’s costing you minutes. Fix those, and even with today’s traversal logic you get a huge step‑function improvement.

---

## Optional (after the fundamentals)

* **Two‑stage parent search:** do parent selection in whitened Euclidean (float32) and only evaluate the residual kernel on the top‑L candidates (with a provable bound or a conservative audit). You already have the whitened cache for the grid—this is the same idea applied to parent search. It’s a bigger design change, but it removes most residual kernel calls from traversal.
* **`argpartition` everywhere:** wherever you still fully sort, switch to selection.
* **Parallel Numba kernel:** if your chunk kernel is single‑threaded, add `parallel=True` + `prange` over query tiles. Keep MKL at 1 thread to avoid oversubscription.

---

## TL;DR / Action plan

1. Replace `kernel_provider` with a **float32 GEMM‑based** tile (no broadcasting).
2. **Ban** kernel recomputation in conflict graph; **require** the cached pairwise block.
3. Normalize all residual‑path arrays to **float32, C‑contiguous** once at configuration time.
4. Swap full sorts for `argpartition`.
5. Eliminate Python callouts inside Numba (decoder, helpers).
6. Set sane thread env vars to avoid oversubscription.

These are mechanical changes, not speculative gating tweaks; they attack the big rocks your telemetry already points to. If you want, I can draft the exact diff stubs for `configure_residual_correlation(...)`, a `make_rbf_provider(...)`, and the `build_conflict_graph(...)` assertion/parameter threading in your codebase structure.
