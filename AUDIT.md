You’ve already squeezed a lot of juice out of the obvious places (scopes → adjacency → MIS, CSR on‑device, Numba everywhere, chunk clamps, caching). If the goal is to *meaningfully* pull 32k builds down (without sacrificing the very fast queries), I think you need a couple of fresh moves that change the work you do—not just how fast you do it.

Below are **twelve new angles** ordered roughly by *impact vs. effort* for your codebase as it stands. Each comes with (a) why it should help *given your telemetry*, (b) how to drop it into your current modules/flags, and (c) what to watch for in tests.

---

## A. Replace (part of) MIS with *leader election* that doesn’t need the full conflict graph

### 1) Hash‑grid leader election (unit‑disk MIS surrogate)

**Why:** Your dominated levels are near‑cliques; building & solving an MIS there wastes time. For Euclidean metrics, a **shifted hash‑grid** with cell width (2^{\ell}) (or ((1-\epsilon)2^{\ell})) elects one “leader” per cell via a deterministic priority (e.g., 64‑bit hash of point id). This is a classic surrogate for MIS on unit‑disk graphs; run a few random *shifts* (2–4) to kill boundary artifacts, take the union, then do a tiny local MIS only among adjacent cells to restore maximality.

**How (drop‑in):**

* `algo/conflict_graph.py`: add `impl="grid"` (behind `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`) that **skips adjacency**:

  1. Bin batch points by (\lfloor x / 2^{\ell}\rfloor) (int grid coords; support multiple random shifts).
  2. For each occupied cell, pick the min‑priority point (priority = 64‑bit mix of (point_id, level, seed)).
  3. Optionally check immediate neighbors’ leaders (3×3 Moore) and run a **tiny** Luby MIS just on that induced graph (≪1% of the full density).
* `algo/mis.py`: reuse existing Luby for the micro‑graph.
* `algo/traverse.py`: nothing to change.
* Telemetry: emit `grid_cells`, `leaders_before_local_mis`, `local_mis_nodes/edges`, `coverage_ratio`.

**Tests:** Verify separation (= cover radius) and maximality at level ( \ell ) by brute‑force within 1–2 rings of each leader cell. Your Tier‑B test that asserts per‑level separation should pass; add one asserting that local MIS restores maximality.

**Expected win:** On dominated batches the heavy scope→CSR→MIS block collapses to O(batch) binning + tiny MIS. In practice this has been a 5–20× build reduction on dense annuli.

---

### 2) Two‑phase MIS: **coarse leaders** then **correct locally**

**Why:** Same spirit as (1) but geometry‑agnostic. Phase A picks a sparse set via a cheap rule; Phase B runs MIS *only* where Phase A leaves conflicts.

**How:**

* Phase A: assign a random priority vector and accept any node that beats all neighbors *within a bounded neighborhood you can fetch without CSR*, e.g., via `scope_indices` + **counting‑sort per parent id**, not full adjacency. (You already have segment hashing and per‑parent semisort—reuse it.)
* Phase B: build CSR **only** for rejected cells/clusters (expect ≪ total).

**Expected win:** Keeps your correctness story while avoiding global CSR on dense levels.

---

## B. Change the *batch* you present to the builder

### 3) **Hilbert sort** new points before prefix‑doubling

**Why:** Your “domination ratio ≈ 0 → mega‑scopes” is a symptom of mixing far‑apart points in one prefix. Sorting the incoming batch by a space‑filling curve (Hilbert/Morton in dim=8) makes annuli sparser and scopes smaller.

**How:**

* `benchmarks.batch_ops` / your build job: before scheduling the prefix, sort `insert_batch` by a Hilbert index (int64) computed from scaled coordinates.
* Keep a feature flag: `COVERTREEX_BATCH_ORDER={random|hilbert}`.

**Expected win:** Often halves clique densities; combined with leader‑election, it compounds.

---

### 4) **Adaptive prefix growth** driven by live domination ratio

**Why:** Your prefix‑doubling is blind to density. Let the **per‑level domination ratio** and `conflict_adj_pairs` decide the next prefix size. If density is high, grow slower (avoid flooding one dominated batch); if low, grow faster.

**How:**

* `algo/batch_insert.py`: compute `rho = dominated / batch_size` after traversal; if `rho > τ_hi`, set `next_prefix = curr + α_small * curr`; else `α_large`.
* Telemetry: write `prefix_factor`, `rho` per batch to the JSONL you already have.

**Expected win:** Smooths the “first dominated batch spike” that currently produces 1.9k shards and 300+ ms scatter.

---

## C. Avoid building a *global* conflict graph

### 5) **Parent‑centric local MIS** only

**Why:** With compressed cover semantics, conflicts that matter are mostly **siblings under nearby parents** at (\ell). You already have per‑parent semisort buffers; you can run MIS **per parent** (plus 1‑ring neighbors) and stitch the results. This caps clique size by local tree degree, not by batch size.

**How:**

* During traversal, let `scope_indices` retain the “owning parent id” for each membership entry.
* Group memberships by parent id (you already have `_group_by_key_counting`).
* For each parent group (and optionally immediate neighbor parents; 1‑ring via `Next`), run the Numba MIS or the grid leader‑election shortcut; no global CSR.
* Keep a rare **spill** path: if a group exceeds `group_cap` after dedupe, fall back to your existing dense builder *for that group only*.

**Expected win:** Turns O(b²) worst‑case into sum of small MIS instances; big win where your logs show `conflict_scope_groups_unique=1` (a single massive scope).

---

## D. Change “what distance we evaluate” in residual mode

### 6) Residual **two‑gate**: cheap bound first, exact only near boundary

**Why:** You already have an early‑exit residual kernel. Make it a formal **cascade**:

1. **Whitened Euclidean gate** (cheap): pre‑whiten `v_matrix` columns once so (|v_i - v_j|_2) becomes a **Lipschitz upper bound** on your residual distance. Reject obviously‑far pairs here using float32.
2. Only for survivors compute the tight residual with the numba kernel.

**How:**

* `metrics/residual.py`: at `configure_residual_correlation`, compute a Cholesky/diagonal rescale for `v_matrix` offline (host once). Cache `v32` and `norms32`.
* `compute_residual_distances_with_radius`: do **gate‑1** on tiles (float32) with a safety margin; then call existing `_distance_chunk` only on survivors.

**Expected win:** On your residual 32k runs the annulus is near‑dense; a 2–3× reduction in pair evaluations is realistic, and it composes with grid/MIS changes.

---

### 7) Residual **landmark lower bounds** (triangle inequality on correlation)

**Why:** Precompute dot‑products to 8–16 landmarks; use them to bound the attainable correlation via Hölder/Cauchy–Schwarz and reject before the heavy loop.

**How:**

* Precompute `L×N` dot table in float32; keep per‑point norms.
* In `_distance_chunk` start with **max possible correlation** bound from landmarks; if it’s already below threshold, skip.

**Expected win:** Similar to (6); often cheaper than whitening if you can afford the small L×N memory.

---

## E. Make traversal **sparser by construction**

### 8) **Skip Next‑chain expansion** with a cap + deferral

**Why:** Your logs show Next‑chain expansion is already cheap, but on dominated levels it still inflates scopes. Instead, **cap expansion length** per query to a small number (e.g., 4), and push remainders into a **deferred queue** handled only if MIS kept the candidate parent.

**How:**

* `traverse_collect_scopes`: emit `(parent, partial_scope)` + a “continuation token”.
* After MIS picks survivors, revisit only survivors with their tokens to finish the chain.

**Expected win:** Reduces scope sizes *before* the conflict stage; costs a small post‑MIS pass (but only over survivors).

---

### 9) **Tile‑wise pair enumeration** with *degree caps*

**Why:** You’ve planned tiles; add a **degree cap per node** in tile expansion (keep ≤K nearest candidates in the annulus per node using a small heap) *before* MIS. MIS only needs an independent set; very long degree tails don’t help.

**How:**

* `_expand_pairs_directed`: early‑select top‑K by annulus distance per source (K=32–64), store only those; expose `avg_degree_before/after_cap` in telemetry.
* Gate it behind `COVERTREEX_DEGREE_CAP`.

**Expected win:** Roughly linear in the degree‑tail heaviness; often 2–5× in dense shells with minimal effect on maximality after local correction.

---

## F. Scheduling & memory

### 10) **Degree‑aware shard merging** (what you proposed—make it explicit)

**Why:** You already noticed: merging shards should use **candidate pair counts**, not just membership. Concretely, choose merges that minimize *(\sum \binom{m_c}{2})* over shards.

**How:**

* Inside `_chunk_ranges_from_indptr`, maintain approximate `m_c` per shard (you can get this *without* materializing pairs via frequency tables).
* Greedily merge tails if `Δ pairs` is below a threshold; expose `pairs_before/after`, not just `members_*`.

**Expected win:** Turns your first dominated batch from 1.9k shards / 364 ms scatter into a few hundred shards / <100 ms scatter.

---

### 11) **Batch re‑use of per‑level CSR buffers** (arena style)

**Why:** On 32k you still report multi‑GiB high‑water RSS. Recycle `sources/targets/indptr/indices` arenas sized by a *moving max* across prior batches; zero only the heads.

**How:**

* `algo/_scope_numba.py`: introduce a small Python‑side pool object whose raw arrays are passed into the Numba kernels (Numba accepts array views). Reset with a fill kernel (`nb.prange` zero heads) instead of realloc.

**Expected win:** Drops allocator chatter and keeps scatter stable; small wall gains but big stability improvement.

---

## G. If you can bend exactness a bit

### 12) **Approximate maximality with micro‑repair**

**Why:** “Maximal independent set” can be approximated quickly; a **single** Luby round with high‑quality priorities gives you a large independent set. Then run a *micro‑repair* that tries to insert any remaining node whose neighbors aren’t chosen, checking only its small local neighborhood.

**How:**

* In `mis.run_mis`, add mode `fast`: one or two Luby iterations, then `repair`: iterate rejected nodes in random order and insert if none of its (at‑most‑K) neighbors are in (use your per‑parent grouping to cap K).
* Tie that to a tolerable separation slack (e.g., accept one violation per 10k nodes and repair deterministically).

**Expected win:** 2–10× on worst graphs. Your Tier‑B invariants will flag if you miss repairs—good guardrail.

---

# A short “what to try this week” plan

1. **Grid leader election (Euclidean)**
   Implement the `grid` conflict‑graph impl and compare against today’s dense path on:

   * 8,192 / 1,024 / k=16 and 32,768 / 1,024 / k=8.
     Log: build wall, `leaders`, `local_mis_nodes`, and coverage.
     *Goal:* cut dominated‑batch time by **≥5×** with identical separation/maximality.

2. **Hilbert‑sorted batches + adaptive prefix**
   Add `--batch-order hilbert` to the CLI and `prefix_factor` adaptation.
   *Goal:* first dominated batch `conflict_adj_scatter_ms < 80 ms` without chunking; smoother per‑batch timings.

3. **Residual two-gate**
   Add float32 whitened gate in residual metric.
   *Goal:* reduce residual `_distance_chunk` call count by **≥50%**; measure `residual_pruned_by_gate1`.
   *Status 2025‑11‑08:* the offline profile + lookup path landed. Run `python tools/build_residual_gate_profile.py docs/data/residual_gate_profile_diag0.json --tree-points 2048 --dimension 8 --seed 42 --bins 512` to refresh the diag0 curve (2,096,128 samples). At runtime, point `COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH` at that file, set `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1`, and turn the gate on via `COVERTREEX_RESIDUAL_GATE1=1` (and optional `..._LOOKUP_MARGIN`, `..._RADIUS_CAP`). Audit mode stays enabled by default when we experiment so we still fail fast if the lookup is too aggressive. The gate remains opt-in for now—we need sparse traversal enabled before the lookup actually prunes anything—but the calibration story is no longer blocked on manual α/margin sweeps.

If those three land, re‑run your 32k tables—*before* touching the sparse traversal kernel—that alone should push Euclidean build well under ~15–20 s and residual under ~30–40 s on CPU, while preserving your current query wins.

---

## Pseudocode sketches (drop‑in friendly)

### Grid leader election (per level)

```python
# covertreex/algo/conflict_graph_builders.py
def build_conflict_grid(level, points_batch, radii, *, shifts=3, seed=42):
    cell_w = 2.0 ** level
    leaders = set()
    for s in range(shifts):
        off = hash64(seed + s) & ((1<<32)-1)
        # integer grid coords for each point
        gx = np.floor((points_batch + off) / cell_w).astype(np.int64)
        # key = tuple grid cell; priority = 64b mix of (point_id, level, s)
        keys = gx.view([('c'+str(d), gx.dtype) for d in range(gx.shape[1])]).ravel()
        prio = mix64(point_ids, level, s)
        # counting-sort by keys, then pick argmin prio per key
        cell_head = argmin_by_key(keys, prio)  # O(n)
        leaders.update(cell_head)
    # Optional: local MIS among neighbor-cell leaders only
    neighbor_edges = edges_among_neighbor_cells(leaders, gx, cell_w, radii)
    S = luby_mis_small(neighbor_edges, leaders)  # tiny graph
    return np.fromiter(S, dtype=np.int64)
```

### Adaptive prefix growth

```python
# batch_insert.py (after traversal)
rho = dominated_count / batch_size
if rho > cfg.high_density:     next_prefix = int(curr + cfg.alpha_small * curr)
elif rho < cfg.low_density:    next_prefix = int(curr + cfg.alpha_large * curr)
else:                           next_prefix = int(curr + cfg.alpha_mid   * curr)
next_prefix = min(next_prefix, max_batch)
```

### Residual two‑gate

```python
# metrics/residual.py
def residual_gate1_whitened(v32, norms32, i, cand_idx, radius, eps=1e-6):
    # fast float32 bound (pre-whitened)
    # derive a conservative mapping from L2 difference to residual radius
    dist32 = l2_chunk(v32[i], v32[cand_idx])              # prange
    # alpha calibrates bound safety; keep if might pass residual threshold
    keep = dist32 <= alpha(radius) + eps
    return keep

def compute_residual_distances_with_radius(...):
    gate = residual_gate1_whitened(backend.v32, backend.n32, query_index, chunk_indices, radius)
    survivors = chunk_indices[gate]
    if survivors.size == 0:
        return np.empty(0), np.zeros(0, dtype=np.uint8)
    return _distance_chunk(... only survivors ...)
```

---

## Micro‑optimizations you can bank today

* **Degree cap in `_expand_pairs_directed`** (see idea #9): keep top‑K annulus neighbors per node before writing pairs; exposes `avg_degree_capped`.
* **int32 everywhere** for indices/ids on ≤32k workloads (you already do some; audit the residual path and CSR heads/tails).
* **Arena reuse** for CSR and scatter buffers (idea #11).
* **Numba signatures**: explicitly declare array dtypes/shapes in hot kernels to avoid shape‑based recompiles across batches (a common hidden tax).

---

## What *not* to chase next (low ROI for your profile)

* Global GraphBLAS/graph‑coloring MIS on CPU: great library work, but your bottleneck today is *building* the graph and the batch schedule, not solving MIS once you’ve shrunk the graph.
* More tiling in the current dense builder **without** changing selection: you’ve already pushed CSR/scatter to sub‑millisecond on steady state; the remaining win is reducing *how many candidates reach it*.

---

If you want, I can draft a minimal PR skeleton for **`impl="grid"`** (builder + tests + flags) and a one‑pager on the **residual two‑gate** interfaces—those two alone should give you fresh headroom while keeping the rest of your architecture intact.
