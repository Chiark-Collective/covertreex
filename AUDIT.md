You're right: with the current code you can make the build fast **or** the queries fast, but the defaults aren’t yet arranged so you get *both* in one path.

Below is a concrete, implementation‑level plan to land a configuration and a few small code changes that **prioritize build time** and, *given that*, deliver near‑best query time—without branching into fundamentally different code paths or reintroducing GPU/JAX overhead.

---

## TL;DR (what to do now)

1. **Go all‑in on the CPU+Numba pipeline end‑to‑end** (you already forced CPU in config): use the Numba scope/adjacency builder *and* the Numba k‑NN walker by default.
2. **Make “fast‑build” the shipped default profile** and keep it balanced for queries:

   * Chunked, deduped scope → radius‑pruned adjacency (Numba) → histogram‑CSR (no argsort) → Luby MIS.
   * After each batch, run a **lightweight “finalize_for_query” step** (O(n)) that materialises a compact `NumbaTreeView` for the k‑NN path.
3. **Turn three knobs adaptively per batch** to keep build bounded:

   * **Batch size** via a pairs budget.
   * **Conflict graph impl** via `auto` (dense/Numba until pairs budget exceeded, segmented otherwise).
   * **Degenerate fast‑path**: switch to sequential insertion when MIS would keep ≤1.
4. **Use the Numba k‑NN by default** with tiny, query‑only buffers—no Python loops, no device conversions, cached child chains and distances.

This combination gives you the fastest build you’ve measured so far *and* the sub‑millisecond per‑query latencies you’ve already seen from the Numba walker, in one consistent runtime.

---

## “Golden path” defaults (production)

Set these as the default profile your API chooses unless the user explicitly asks otherwise.

```bash
# Runtime
COVERTREEX_DEVICE=cpu:0
COVERTREEX_ENABLE_NUMBA=1
COVERTREEX_PRECISION=float64     # keep FP64 as required by Vecchia

# Conflict graph
COVERTREEX_CONFLICT_GRAPH_IMPL=auto     # see heuristic below
COVERTREEX_SCOPE_SEGMENT_DEDUP=1
COVERTREEX_SCOPE_CHUNK_TARGET=65536     # good L2-sized default on modern CPUs

# Diagnostics
COVERTREEX_ENABLE_DIAGNOSTICS=0         # keep off for throughput
COVERTREEX_LOG_LEVEL=warning
```

In code, expose the same “fast‑build” profile:

```python
tree = PCCTree.build(points, profile="fast-build")  # applies the knobs below
```

---

## 1) Keep the **fastest build path** by default

You already have the pieces; unify them behind `profile="fast-build"`:

### 1.1 Scope → adjacency pipeline (stay Numba, avoid argsort)

* **Scope grouping:** keep your chunked, Numba path with segment‑level hashing and counting‑sort grouping. This removed the previous 150–250 ms hotspot per dominated batch—retain it.
* **Radius‑pruned expansion:** continue writing **directed** edges inside the Numba kernel while checking `dist² ≤ min(r_i², r_j²)` in the expansion loop. (You’ve landed this; it’s the right place for the filter.)
* **Histogram CSR assembly (replace argsort):**

  * Precount out‑degree with `np.bincount(sources, minlength=batch_size)`.
  * Build `indptr` by prefix sum.
  * Fill `indices` by a single pass that uses a thread‑local cursor array.
  * *Do not* sort per‑row unless a consumer requires it; MIS does not.

  Sketch:

  ```python
  # sources, targets: directed edges written by the Numba builder
  deg = np.bincount(sources, minlength=batch_size).astype(np.int32)
  indptr = np.empty(batch_size + 1, np.int32)
  indptr[0] = 0
  np.cumsum(deg, out=indptr[1:])
  # row_cursors starts as a copy of indptr
  row_cursors = indptr[:-1].copy()
  indices = np.empty_like(sources, dtype=np.int32)
  for s, t in zip(sources, targets):
      pos = row_cursors[s]
      indices[pos] = t
      row_cursors[s] = pos + 1
  # Optional: unique per row only if MIS impl requires it (usually not)
  ```

  This removes the `argsort` hotspot you noted and keeps build time flat as n grows, until the pairs budget (next point) is hit.

### 1.2 Pairs budget + adaptive batch size

Keep build fast by bounding work per batch:

* Compute an **upper bound on pairs per batch** from scope group sizes you already have:

  [
  \text{pairs} = \sum_g m_g (m_g-1)
  ]

* Choose `batch_size` (prefix‑doubling cap) so `pairs ≤ P_budget`. Good defaults:

  * `P_budget = 4–8 × 10^6` directed pairs per batch on a 16–32‑core workstation.
  * `batch_size_cap = 256` for n ≤ 8 k, 128 for larger n unless the scope histogram shows low co‑occurrence.

* If the observed `pairs` for the current batch exceeds the budget, **halve** the batch and retry (you already have the prefix structure to do this deterministically).

This one heuristic prevents an O(b²) cliff and keeps your new Numba path in the “few ms per batch” zone.

### 1.3 Degenerate‑batch fast path

You have this sketched—turn it on by default:

* If MIS would keep ≤1 anchor in a level (use your domination ratio or a quick greedy pass), skip conflict‑graph/MIS and run **sequential insertion** for that subset immediately.

On your dominated quick runs this skips all the machinery and wins ~hundreds of ms per batch without touching correctness.

### 1.4 MIS: keep Luby, fuse “finalize”

MIS is already a rounding error for you. Two tiny tweaks keep it that way:

* Use **int32** CSR throughout MIS to reduce memory bandwidth.
* After Luby’s parallel rounds, run a *single* greedy finalize over remaining vertices to guarantee maximality; this removes rare long tails at high degrees with negligible cost.

---

## 2) Make queries fast without hurting build

The trick is to *not* slow the builder with query‑time conveniences, and instead do a cheap “finalize” after each batch.

### 2.1 Post‑batch `finalize_for_query()` (O(n), cache‑friendly)

Right after each batch is applied, **materialise a query view** in contiguous arrays:

* `child_starts: int32[n+1]`, `child_indices: int32[num_edges]` (CSR children).
* `next_cache: int32[n]`, already present—pack it into the same SoA layout.
* `si_cache: int32[...]` and per‑node radii **squared** (`float64[n]`).
* Optional: a **BFS‑order renumbering** into a dense `[0..n)` “query id” space so child lists and `next` chains are cache‑friendly.

Export this as a `NumbaTreeView` and keep it alongside the immutable `PCCTree` version. The finalize step is linear and measured in milliseconds at your scales; it gives the Numba walker everything it needs without paying costs during the build.

### 2.2 Default to the **Numba k‑NN walker**

* Always use `queries._knn_numba` when `ENABLE_NUMBA=1`. It already:

  * Reuses distances,
  * Uses precomputed child lists,
  * Keeps deterministic tie‑breaks.
* Keep k‑selection **heapless** for small `k` (≤32): maintain a fixed‑size top‑k array plus running max instead of a Python heap; guard with a tiny templated helper in Numba.
* **Leaf dense fallback:** when a local frontier exceeds a threshold (e.g., >512 candidates) or the cover radius drops below a small ε, switch to a dense distance pass on that frontier and finish with a stable `argpartition`. This improved your early prototype to ~30 ms; re‑enable it with a low threshold so it rarely triggers.

These keep the query tight (≈0.1–0.2 ms/query at your 2 k/8 k sizes in your notes) without back‑pressuring the builder.

---

## 3) One runtime, two internal “impls” with **auto** switch

Keep a single public path but switch internals automatically:

```text
COVERTREEX_CONFLICT_GRAPH_IMPL=auto
└── if pairs_per_batch <= P_budget and batch_size <= 256:
      use numba_dense (chunked, deduped, radius-pruned, histogram CSR)
    else:
      use segmented (CSR-by-scope)  # avoids dense membership blow-up
```

This avoids regressing large‑n builds while keeping small/medium builds extremely fast.

---

## 4) A few small correctness‑neutral optimizations

All cheap, all measurable:

* **Use squared distances & radii everywhere** (you already do in places). Don’t take square roots in the builder or walker.
* **Indices as int32** end‑to‑end (CSR + MIS + k‑NN). You’ve already moved many arrays; make it uniform.
* **Precompute norms** per point once (Numba) and use `‖x−y‖² = ‖x‖²+‖y‖²−2x·y` in dense fallbacks. It shaved ms in your earlier dense path.
* **Warm the Numba caches**: run a 1‑query dry call in `PCCTree.from_points()` (guarded by a flag) so first “real” queries don’t pay compilation. (This doesn’t change throughput; it just amortizes the first‑call blip you observed.)

---

## 5) Instrumentation that matters for the auto‑knobs

Keep diagnostics light but preserve the counters that drive the adaptive logic:

* `conflict_adj_pairs`, `conflict_adj_max_group`, `scope_groups`, `scope_groups_unique`, `scope_domination_ratio`, and **wall ms** for `scope_group`, `adjacency_filter`, `csr_build`, `mis`.
* Log **pairs_per_batch** and the selected `batch_size_cap` so you can verify the auto‑scaler behaves. Suppress per‑pair logs by default.

---

## 6) What this yields (based on your current snapshots)

* On 2 048/512/k=8 with Numba enabled and diagnostics off you’ve already seen:

  * Build ≈ **10–15 s** (depending on which iteration), Queries ≈ **0.10–0.12 s** total (≈0.2 ms/query).
* On 8 192/1 024/k=16:

  * Build ≈ **23–32 s**, Queries ≈ **0.93–1.0 s** total (≈1 ms/query) once the histogram‑CSR and pairs budget are in.
* The remaining variability was from `argsort` in CSR and degenerate batches paying full pipelines. The histogram CSR and degenerate fast‑path remove those outliers.

These are within striking distance of the sequential baseline query times while keeping a *parallel* build that’s orders of magnitude faster than the external baselines you logged.

---

## 7) Minimal code changes to land

1. **CSR histogram builder** (replace per‑batch argsort as above).
2. **Auto batch‑size & impl switch** driven by `pairs_per_batch` and `P_budget`.
3. **Finalize‑for‑query**: add `tree.materialise_for_numba()` in the builder and call it after each batch apply.
4. **Default profile plumbing**: `profile="fast-build"` applies the env knobs, enables Numba MIS and k‑NN, sets `CONFLICT_GRAPH_IMPL=auto`, and turns on degenerate fast‑path.
5. **Leaf dense fallback** in `_knn_numba` with a small `frontier_dense_threshold` (configurable).

All of these are local and do not change external semantics.

---

## 8) Suggested defaults (you can tune later)

* `P_budget = 6_000_000` directed pairs/batch.
* `batch_size_cap = 256` if n ≤ 16 k else 128.
* `frontier_dense_threshold = 512`.
* `scope_chunk_target = 65_536` (leave exposed as env).
* `int dtype = int32`, `float dtype = float64`.

---

## 9) Sanity & safety checks to keep green

* Keep your **Tier‑A/B invariants** on by default in CI (not in prod): separation, nesting, persistence diff.
* Add two **regression micro‑bench** tests to catch performance cliffs:

  1. Verify `conflict_adj_pairs / batch` stays ≤ `P_budget` for the quick run.
  2. Assert `csr_build_ms` never exceeds, say, 25 ms on the quick run.

---

### Why this works

* The builder stays dominated by *linear* and *chunked quadratic* work with a strict pairs budget, no global sorts.
* The query path becomes a thin, branch‑predictable Numba loop over contiguous arrays, warmed once, with distance reuse.
* The only time you “lose” is when scopes explode; the budget + degenerate fast‑path cut those off early.

If you want, I can translate the CSR histogram and the finalize‑for‑query helper into concrete code against your current module layout.
