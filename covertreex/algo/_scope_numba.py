from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba as nb

    NUMBA_SCOPE_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nb = None  # type: ignore
    NUMBA_SCOPE_AVAILABLE = False

I32 = np.int32
I64 = np.int64
U8 = np.uint8
U64 = np.uint64


@dataclass(frozen=True)
class ScopeAdjacencyResult:
    """Return value for the Numba scope adjacency builder."""

    sources: np.ndarray
    targets: np.ndarray
    max_group_size: int
    total_pairs: int
    candidate_pairs: int
    num_groups: int
    num_unique_groups: int


def _chunk_ranges_from_indptr(indptr: np.ndarray, chunk_target: int) -> list[tuple[int, int]]:
    """Return (start, end) node ranges whose membership volume stays under chunk_target."""

    num_nodes = indptr.size - 1
    if num_nodes <= 0:
        return [(0, 0)]
    if chunk_target <= 0:
        return [(0, num_nodes)]

    ranges: list[tuple[int, int]] = []
    start = 0
    accum = 0
    for node in range(num_nodes):
        accum += int(indptr[node + 1] - indptr[node])
        if accum >= chunk_target:
            ranges.append((start, node + 1))
            start = node + 1
            accum = 0
    if start < num_nodes:
        ranges.append((start, num_nodes))
    if not ranges:
        ranges.append((0, num_nodes))
    return ranges


def _require_numba() -> None:
    if not NUMBA_SCOPE_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError(
            "Numba scope helpers requested but `numba` is not available. "
            "Install the extra '[numba]' extras or disable the feature via "
            "COVERTREEX_ENABLE_NUMBA=0."
        )


if NUMBA_SCOPE_AVAILABLE:

    @nb.njit(cache=True)
    def _membership_point_ids_from_indptr(indptr: np.ndarray, total: int) -> np.ndarray:
        n_points = indptr.size - 1
        out = np.empty(total, dtype=I32)
        for p in range(n_points):
            s = indptr[p]
            e = indptr[p + 1]
            for i in range(s, e):
                out[i] = p
        return out

    @nb.njit(cache=True)
    def _group_by_key_counting(keys: np.ndarray, vals: np.ndarray, K: int):
        counts = np.zeros(K, dtype=I64)
        N = keys.size
        for i in range(N):
            counts[keys[i]] += 1

        indptr = np.empty(K + 1, dtype=I64)
        indptr[0] = 0
        for k in range(K):
            indptr[k + 1] = indptr[k] + counts[k]

        out = np.empty(N, dtype=vals.dtype)
        heads = indptr[:-1].copy()
        for i in range(N):
            k = keys[i]
            j = heads[k]
            out[j] = vals[i]
            heads[k] = j + 1
        return indptr, out

    @nb.njit(cache=True, parallel=True)
    def _sort_segments_inplace(values: np.ndarray, indptr: np.ndarray):
        m = indptr.size - 1
        for i in nb.prange(m):
            s = indptr[i]
            e = indptr[i + 1]
            if e - s > 1:
                tmp = np.sort(values[s:e])
                values[s:e] = tmp

    @nb.njit(cache=True, parallel=True)
    def _hash_segments(values: np.ndarray, indptr: np.ndarray) -> np.ndarray:
        m = indptr.size - 1
        hashes = np.empty(m, dtype=U64)
        for i in nb.prange(m):
            s = indptr[i]
            e = indptr[i + 1]
            h = U64(0xCBF29CE484222325)
            for j in range(s, e):
                v = U64(np.int64(values[j]))
                h ^= v + U64(0x9E3779B97F4A7C15)
                h *= U64(0x100000001B3)
            hashes[i] = h
        return hashes

    @nb.njit(cache=True)
    def _segments_equal(values: np.ndarray, indptr: np.ndarray, a: int, b: int) -> bool:
        sa = indptr[a]
        ea = indptr[a + 1]
        sb = indptr[b]
        eb = indptr[b + 1]
        if (ea - sa) != (eb - sb):
            return False
        L = ea - sa
        for i in range(L):
            if values[sa + i] != values[sb + i]:
                return False
        return True

    @nb.njit(cache=True)
    def _dedupe_segments_by_hash(
        values: np.ndarray, indptr: np.ndarray, hashes: np.ndarray
    ) -> np.ndarray:
        m = hashes.size
        order = np.argsort(hashes)
        keep = np.zeros(m, dtype=U8)
        i = 0
        while i < m:
            j = i + 1
            ref = order[i]
            keep[ref] = 1
            while j < m and hashes[order[j]] == hashes[ref]:
                cur = order[j]
                if not _segments_equal(values, indptr, ref, cur):
                    keep[cur] = 1
                j += 1
            i = j
        return keep.view(np.bool_)

    @nb.njit(cache=True)
    def _compute_pair_counts(indptr: np.ndarray, keep: np.ndarray):
        m = keep.size
        total_pairs = I64(0)
        pair_counts = np.empty(m, dtype=I64)
        max_group = 0
        for i in range(m):
            if keep[i]:
                c = indptr[i + 1] - indptr[i]
                if c > 1:
                    pc = (c * (c - 1)) // 2
                    pair_counts[i] = pc
                    total_pairs += pc
                    if c > max_group:
                        max_group = c
                else:
                    pair_counts[i] = 0
            else:
                pair_counts[i] = 0
        return pair_counts, total_pairs, max_group

    @nb.njit(cache=True)
    def _prefix_sum(arr: np.ndarray):
        out = np.empty(arr.size + 1, dtype=I64)
        s = I64(0)
        out[0] = 0
        for i in range(arr.size):
            s += arr[i]
            out[i + 1] = s
        return out

    @nb.njit(cache=True, parallel=True)
    def _expand_pairs(
        values: np.ndarray,
        indptr: np.ndarray,
        keep: np.ndarray,
        offsets: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
    ):
        m = keep.size
        for i in nb.prange(m):
            if not keep[i]:
                continue
            s = indptr[i]
            e = indptr[i + 1]
            c = e - s
            if c <= 1:
                continue
            k = offsets[i]
            for a in range(c):
                pa = values[s + a]
                for b in range(a + 1, c):
                    pb = values[s + b]
                    src[k] = pa
                    dst[k] = pb
                    k += 1

    @nb.njit(cache=True)
    def _dedup_pairs_undirected(src: np.ndarray, dst: np.ndarray):
        E = src.size
        keys = np.empty(E, dtype=U64)
        for i in range(E):
            a = src[i]
            b = dst[i]
            if a < b:
                x = a
                y = b
            else:
                x = b
                y = a
            keys[i] = (U64(np.int64(x)) << U64(32)) | (U64(np.int64(y)) & U64(0xFFFFFFFF))
        order = np.argsort(keys)
        uniq = np.ones(E, dtype=U8)
        for i in range(1, E):
            if keys[order[i]] == keys[order[i - 1]]:
                uniq[i] = 0

        count = 0
        for i in range(E):
            if uniq[i]:
                count += 1

        s2 = np.empty(count, dtype=I32)
        t2 = np.empty(count, dtype=I32)
        k = 0
        for i in range(E):
            if uniq[i]:
                idx = order[i]
                a = src[idx]
                b = dst[idx]
                if a < b:
                    s2[k] = a
                    t2[k] = b
                else:
                    s2[k] = b
                    t2[k] = a
                k += 1
        return s2, t2

    @nb.njit(cache=True, parallel=True)
    def _expand_pairs_directed(
        values: np.ndarray,
        indptr: np.ndarray,
        kept_nodes: np.ndarray,
        offsets: np.ndarray,
        pairwise: np.ndarray,
        radii: np.ndarray,
    ):
        k = kept_nodes.size
        capacity = offsets[-1]
        sources = np.empty(capacity, dtype=I32)
        targets = np.empty(capacity, dtype=I32)
        used = np.zeros(k, dtype=I64)

        for idx in nb.prange(k):
            node = int(kept_nodes[idx])
            s = indptr[node]
            e = indptr[node + 1]
            c = e - s
            if c <= 1:
                continue
            base = offsets[idx]
            write = 0
            for a in range(c - 1):
                pa = values[s + a]
                ra = radii[pa]
                for b in range(a + 1, c):
                    pb = values[s + b]
                    rb = radii[pb]
                    bound = ra if ra < rb else rb
                    if pairwise[pa, pb] <= bound:
                        sources[base + write] = pa
                        targets[base + write] = pb
                        write += 1
                        sources[base + write] = pb
                        targets[base + write] = pa
                        write += 1
            used[idx] = write

        return sources, targets, used

    @nb.njit(cache=True)
    def _compact_pairs(
        sources: np.ndarray,
        targets: np.ndarray,
        offsets: np.ndarray,
        used: np.ndarray,
    ):
        total_used = I64(0)
        for i in range(used.size):
            total_used += used[i]
        if total_used == sources.size:
            return sources, targets, int(total_used)

        out_src = np.empty(total_used, dtype=I32)
        out_dst = np.empty(total_used, dtype=I32)
        cursor = I64(0)
        for node in range(used.size):
            count = used[node]
            if count == 0:
                continue
            start_in = offsets[node]
            for j in range(count):
                out_src[cursor + j] = sources[start_in + j]
                out_dst[cursor + j] = targets[start_in + j]
            cursor += count
        return out_src, out_dst, int(total_used)

    def build_conflict_graph_numba_dense(
        scope_indptr: np.ndarray,
        scope_indices: np.ndarray,
        batch_size: int,
        *,
        segment_dedupe: bool = True,
        chunk_target: int = 0,
        pairwise: np.ndarray | None = None,
        radii: np.ndarray | None = None,
    ) -> ScopeAdjacencyResult:
        if scope_indices.size == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                max_group_size=0,
                total_pairs=0,
                candidate_pairs=0,
                num_groups=0,
                num_unique_groups=0,
            )

        if pairwise is None or radii is None:
            raise ValueError(
                "pairwise distances and radii arrays are required for the Numba "
                "conflict-graph builder"
            )

        pairwise_arr = np.ascontiguousarray(np.asarray(pairwise, dtype=np.float64))
        radii_arr = np.asarray(radii, dtype=np.float64)

        total = scope_indices.size
        point_ids = _membership_point_ids_from_indptr(scope_indptr.astype(I64), total)
        num_nodes = int(scope_indices.max()) + 1 if scope_indices.size else 0
        indptr_nodes, node_members = _group_by_key_counting(
            scope_indices.astype(I32),
            point_ids,
            num_nodes,
        )
        _sort_segments_inplace(node_members, indptr_nodes)

        if segment_dedupe:
            hashes = _hash_segments(node_members, indptr_nodes)
            keep = _dedupe_segments_by_hash(node_members, indptr_nodes, hashes)
        else:
            keep = np.ones(indptr_nodes.size - 1, dtype=np.bool_)

        pair_counts, total_pairs, max_group_size = _compute_pair_counts(
            indptr_nodes, keep
        )
        num_groups = int(indptr_nodes.size - 1)
        num_unique_groups = int(np.count_nonzero(keep))
        if total_pairs == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=0,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
            )

        directed_total_pairs = int(total_pairs * 2)
        candidate_pairs = directed_total_pairs
        kept_nodes = np.nonzero(keep)[0].astype(I64)
        if kept_nodes.size == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=candidate_pairs,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
            )
        pair_counts_kept = pair_counts[keep]
        directed_counts = pair_counts_kept * 2
        offsets = _prefix_sum(directed_counts)
        capacity = int(offsets[-1])
        if capacity == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=candidate_pairs,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
            )

        sources, targets, used_counts = _expand_pairs_directed(
            node_members,
            indptr_nodes,
            kept_nodes,
            offsets,
            pairwise_arr,
            radii_arr,
        )
        sources, targets, actual_pairs = _compact_pairs(
            sources,
            targets,
            offsets,
            used_counts,
        )
        if actual_pairs == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=candidate_pairs,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
            )
        return ScopeAdjacencyResult(
            sources=sources,
            targets=targets,
            max_group_size=int(max_group_size),
            total_pairs=actual_pairs,
            candidate_pairs=candidate_pairs,
            num_groups=num_groups,
            num_unique_groups=num_unique_groups,
        )

else:  # pragma: no cover - executed when numba missing

    def build_conflict_graph_numba_dense(
        scope_indptr: np.ndarray,
        scope_indices: np.ndarray,
        batch_size: int,
        *,
        segment_dedupe: bool = True,
        chunk_target: int = 0,
        pairwise: np.ndarray | None = None,
        radii: np.ndarray | None = None,
    ) -> ScopeAdjacencyResult:
        _require_numba()
        raise AssertionError("unreachable")
