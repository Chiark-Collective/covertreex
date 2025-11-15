from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from covertreex.algo.conflict.arena import ScopeBuilderArena

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
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    max_group_size: int
    total_pairs: int
    candidate_pairs: int
    num_groups: int
    num_unique_groups: int
    chunk_count: int = 1
    chunk_emitted: int = 0
    chunk_max_members: int = 0
    chunk_pairs_cap: int = 0
    chunk_pairs_before: int = 0
    chunk_pairs_after: int = 0
    chunk_pair_merges: int = 0
    degree_cap: int = 0
    degree_pruned_pairs: int = 0


@dataclass(frozen=True)
class ChunkRangeStats:
    pair_cap: int = 0
    pair_tail_threshold: int = 0
    pair_max_before: int = 0
    pair_max_after: int = 0
    pair_merges: int = 0


_TAIL_MERGE_DIVISOR = 4


def _chunk_ranges_from_indptr(
    indptr: np.ndarray,
    chunk_target: int,
    max_segments: int = 0,
    keep_mask: np.ndarray | None = None,
    pair_counts: np.ndarray | None = None,
    pair_merge: bool = False,
) -> tuple[list[tuple[int, int]], ChunkRangeStats]:
    """Return (start, end) node ranges bounded by the configured chunk target."""

    num_nodes = indptr.size - 1
    if num_nodes <= 0:
        return [(0, 0)], ChunkRangeStats()

    node_weights = np.diff(indptr).astype(np.int64, copy=False)
    if keep_mask is not None:
        keep_arr = np.asarray(keep_mask, dtype=np.int64)
        if keep_arr.shape[0] != num_nodes:
            raise ValueError("keep_mask shape must match node count")
        node_weights = node_weights * keep_arr

    total_volume = int(node_weights.sum())
    if total_volume <= 0:
        return [(0, num_nodes)], ChunkRangeStats()

    if chunk_target <= 0 and max_segments <= 0:
        return [(0, num_nodes)], ChunkRangeStats()

    if chunk_target <= 0:
        effective_target = max(1, (total_volume + max_segments - 1) // max(1, max_segments))
    else:
        effective_target = max(1, int(chunk_target))
        if max_segments > 0:
            min_target = (total_volume + max_segments - 1) // max_segments
            if min_target > effective_target:
                effective_target = max(1, min_target)

    weights_prefix = np.empty(num_nodes + 1, dtype=np.int64)
    weights_prefix[0] = 0
    np.cumsum(node_weights, out=weights_prefix[1:])

    ranges: list[tuple[int, int]] = []
    start = 0
    accum = 0
    for node in range(num_nodes):
        accum += int(node_weights[node])
        if accum >= effective_target and node + 1 > start:
            ranges.append((start, node + 1))
            start = node + 1
            accum = 0
    if start < num_nodes:
        ranges.append((start, num_nodes))
    if not ranges:
        ranges.append((0, num_nodes))

    raw_ranges = list(ranges)
    if len(ranges) > 1:
        last_start, last_end = ranges[-1]
        last_volume = int(weights_prefix[last_end] - weights_prefix[last_start])
        if last_volume == 0:
            ranges.pop()
            if not ranges:
                return [(0, num_nodes)], ChunkRangeStats()

    if effective_target > 0 and len(ranges) > 1:
        tail_threshold = max(1, (effective_target + _TAIL_MERGE_DIVISOR - 1) // _TAIL_MERGE_DIVISOR)
    else:
        tail_threshold = 0

    pair_stats = ChunkRangeStats()
    pair_prefix = None
    pair_cap = 0
    pair_tail_threshold = 0
    pair_max_before = 0
    pair_max_after = 0
    pair_merge_count = 0
    if pair_counts is not None and pair_counts.size == num_nodes:
        pair_prefix = np.empty(num_nodes + 1, dtype=np.int64)
        pair_prefix[0] = 0
        np.cumsum(pair_counts.astype(np.int64, copy=False), out=pair_prefix[1:])
        if pair_prefix[-1] > 0 and len(raw_ranges) > 0:
            approx_segments = len(raw_ranges)
            pair_cap = max(1, int((pair_prefix[-1] + approx_segments - 1) // approx_segments))
            pair_tail_threshold = max(1, int((pair_cap + _TAIL_MERGE_DIVISOR - 1) // _TAIL_MERGE_DIVISOR))
            pair_max_before = _max_range_pairs(raw_ranges, pair_prefix)

    if effective_target > 0 and len(ranges) > 1:
        ranges = _merge_tail_ranges(
            ranges,
            weights_prefix,
            effective_target,
            tail_threshold,
            pair_prefix=pair_prefix,
            pair_cap=pair_cap,
            pair_tail_threshold=pair_tail_threshold,
        )

    if pair_prefix is not None and pair_prefix[-1] > 0:
        if pair_merge and pair_cap > 0 and len(ranges) > 1:
            ranges, pair_merge_count = _merge_ranges_by_pair_volume(ranges, pair_prefix, pair_cap)
        pair_max_after = _max_range_pairs(ranges, pair_prefix)
        pair_stats = ChunkRangeStats(
            pair_cap=int(pair_cap),
            pair_tail_threshold=int(pair_tail_threshold),
            pair_max_before=int(pair_max_before),
            pair_max_after=int(pair_max_after),
            pair_merges=int(pair_merge_count),
        )
    elif pair_merge_count:
        pair_stats = ChunkRangeStats(pair_merges=int(pair_merge_count))

    return ranges, pair_stats


def _merge_tail_ranges(
    ranges: list[tuple[int, int]],
    weights_prefix: np.ndarray,
    chunk_cap: int,
    tail_threshold: int,
    *,
    pair_prefix: np.ndarray | None = None,
    pair_cap: int = 0,
    pair_tail_threshold: int = 0,
) -> list[tuple[int, int]]:
    if chunk_cap <= 0 or len(ranges) <= 1:
        return ranges

    merged = list(ranges)
    idx = len(merged) - 1
    while idx > 0:
        start, end = merged[idx]
        volume = int(weights_prefix[end] - weights_prefix[start])
        pair_volume = 0
        if pair_prefix is not None:
            pair_volume = int(pair_prefix[end] - pair_prefix[start])
        small_tail = tail_threshold > 0 and volume < tail_threshold
        small_pair_tail = (
            pair_prefix is not None
            and pair_tail_threshold > 0
            and pair_volume < pair_tail_threshold
        )
        if not small_tail and not small_pair_tail:
            idx -= 1
            continue
        prev_start, prev_end = merged[idx - 1]
        prev_volume = int(weights_prefix[prev_end] - weights_prefix[prev_start])
        combined = volume + prev_volume
        if combined == 0:
            merged[idx - 1] = (prev_start, end)
            merged.pop(idx)
            idx -= 1
            continue
        if combined > chunk_cap:
            idx -= 1
            continue
        if pair_prefix is not None and pair_cap > 0:
            combined_pairs = int(pair_prefix[end] - pair_prefix[prev_start])
            if combined_pairs > pair_cap:
                idx -= 1
                continue
        merged[idx - 1] = (prev_start, end)
        merged.pop(idx)
        idx -= 1
    return merged


def _merge_ranges_by_pair_volume(
    ranges: list[tuple[int, int]],
    pair_prefix: np.ndarray,
    pair_cap: int,
) -> tuple[list[tuple[int, int]], int]:
    if pair_cap <= 0 or not ranges or pair_prefix.size == 0:
        return ranges, 0
    merged: list[tuple[int, int]] = []
    merges = 0
    current_start, current_end = ranges[0]
    current_pairs = int(pair_prefix[current_end] - pair_prefix[current_start])
    slack = max(pair_cap // 4, 1)
    cap_with_slack = pair_cap + slack
    for start, end in ranges[1:]:
        next_pairs = int(pair_prefix[end] - pair_prefix[start])
        if current_pairs == 0 and next_pairs == 0:
            current_end = end
            current_pairs = 0
            merges += 1
            continue
        if current_pairs + next_pairs <= cap_with_slack:
            current_end = end
            current_pairs += next_pairs
            merges += 1
            continue
        merged.append((current_start, current_end))
        current_start = start
        current_end = end
        current_pairs = next_pairs
    merged.append((current_start, current_end))
    return merged, merges


def _max_range_pairs(ranges: list[tuple[int, int]], pair_prefix: np.ndarray | None) -> int:
    if pair_prefix is None or pair_prefix.size == 0 or not ranges:
        return 0
    peak = 0
    for start, end in ranges:
        count = int(pair_prefix[end] - pair_prefix[start])
        if count > peak:
            peak = count
    return peak


def _require_numba() -> None:
    if not NUMBA_SCOPE_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError(
            "Numba scope helpers requested but `numba` is not available. "
            "Install the extra '[numba]' extras or disable the feature via "
            "COVERTREEX_ENABLE_NUMBA=0."
        )


_SCOPE_WARMED = False


def warmup_scope_builder() -> None:
    """Trigger Numba compilation for the scope adjacency kernels."""

    global _SCOPE_WARMED
    if _SCOPE_WARMED or not NUMBA_SCOPE_AVAILABLE:
        return

    scope_indptr = np.asarray([0, 3, 5], dtype=np.int64)
    scope_indices = np.asarray([0, 1, 2, 1, 2], dtype=np.int64)
    pairwise = np.zeros((6, 6), dtype=np.float64)
    radii = np.ones(6, dtype=np.float64)

    build_conflict_graph_numba_dense(
        scope_indptr,
        scope_indices,
        batch_size=2,
        segment_dedupe=True,
        chunk_target=0,
        pairwise=pairwise,
        radii=radii,
    )
    build_scope_csr_from_pairs(
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([3, 2, 1], dtype=np.int64),
        2,
        limit=1,
        top_levels=np.asarray([2, 5, 4, 3], dtype=np.int64),
        parents=np.asarray([3, 1], dtype=np.int64),
    )
    _SCOPE_WARMED = True


if NUMBA_SCOPE_AVAILABLE:
    @nb.njit(cache=True)
    def _select_topk_by_index(indices: np.ndarray, limit: int) -> np.ndarray:
        count = indices.size
        if count == 0:
            return np.empty(0, dtype=I64)
        order = np.argsort(indices)
        keep = count if limit <= 0 or limit >= count else limit
        out = np.empty(keep, dtype=I64)
        for i in range(keep):
            out[i] = np.int64(indices[order[i]])
        return out

    @nb.njit(cache=True)
    def _select_topk_by_level(indices: np.ndarray, top_levels: np.ndarray, limit: int) -> np.ndarray:
        count = indices.size
        if count == 0:
            return np.empty(0, dtype=I64)
        levels = np.empty(count, dtype=I64)
        max_level = np.int64(-9_223_372_036_854_775_808)
        size_levels = top_levels.size
        for i in range(count):
            idx_val = int(indices[i])
            level = np.int64(-1)
            if 0 <= idx_val < size_levels:
                level = np.int64(top_levels[idx_val])
            levels[i] = level
            if level > max_level:
                max_level = level
        shift = np.int64(1) << np.int64(32)
        keys = np.empty(count, dtype=np.int64)
        for i in range(count):
            key_level = max_level - levels[i]
            if key_level < 0:
                key_level = 0
            keys[i] = key_level * shift + np.int64(indices[i])
        order = np.argsort(keys)
        keep = count if limit <= 0 or limit >= count else limit
        out = np.empty(keep, dtype=I64)
        for i in range(keep):
            out[i] = np.int64(indices[order[i]])
        return out

    @nb.njit(cache=True)
    def _contains_value(values: np.ndarray, needle: int) -> bool:
        if needle < 0:
            return False
        for i in range(values.size):
            if int(values[i]) == needle:
                return True
        return False

    @nb.njit(cache=True)
    def _build_scope_csr_from_pairs_impl(
        owners: np.ndarray,
        members: np.ndarray,
        num_nodes: int,
        limit: int,
        top_levels: np.ndarray,
        parents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        indptr = np.zeros(num_nodes + 1, dtype=I64)
        if owners.size == 0 or num_nodes <= 0:
            return indptr, np.empty(0, dtype=I32)

        counts = np.zeros(num_nodes, dtype=I64)
        for i in nb.prange(owners.size):
            node = int(owners[i])
            if 0 <= node < num_nodes:
                counts[node] += 1

        for idx in range(num_nodes):
            indptr[idx + 1] = indptr[idx] + counts[idx]

        total = int(indptr[-1])
        indices = np.empty(total, dtype=I32)
        cursors = indptr[:-1].copy()
        for i in nb.prange(owners.size):
            node = int(owners[i])
            if node < 0 or node >= num_nodes:
                continue
            pos = cursors[node]
            if pos >= indptr[node + 1]:
                continue
            indices[pos] = members[i]
            cursors[node] = pos + 1

        limit_value = int(limit) if limit > 0 else 0
        have_levels = top_levels.size > 0
        have_parents = parents.size == num_nodes

        if limit_value <= 0 and not have_levels:
            return indptr, indices

        trimmed_counts = np.zeros(num_nodes, dtype=I64)
        for node in range(num_nodes):
            start = indptr[node]
            end = indptr[node + 1]
            count = end - start
            if count == 0:
                continue
            window = np.empty(count, dtype=I64)
            for j in range(count):
                window[j] = np.int64(indices[start + j])
            selected: np.ndarray
            if have_levels:
                selected = _select_topk_by_level(window, top_levels, limit_value)
            else:
                selected = _select_topk_by_index(window, limit_value)

            if (
                limit_value > 0
                and have_levels
                and have_parents
                and node < parents.size
            ):
                parent_idx = int(parents[node])
                if parent_idx >= 0 and not _contains_value(selected, parent_idx):
                    augmented = np.empty(selected.size + 1, dtype=I64)
                    for j in range(selected.size):
                        augmented[j] = selected[j]
                    augmented[selected.size] = np.int64(parent_idx)
                    selected = _select_topk_by_level(augmented, top_levels, limit_value)

            keep = selected.size
            trimmed_counts[node] = keep
            for j in range(keep):
                indices[start + j] = np.int32(selected[j])

        total_trimmed = int(np.sum(trimmed_counts))
        if total_trimmed == 0:
            return np.zeros(num_nodes + 1, dtype=I64), np.empty(0, dtype=I32)

        new_indptr = np.empty(num_nodes + 1, dtype=I64)
        new_indptr[0] = 0
        for node in range(num_nodes):
            new_indptr[node + 1] = new_indptr[node] + trimmed_counts[node]

        new_indices = np.empty(total_trimmed, dtype=I32)
        for node in range(num_nodes):
            keep = int(trimmed_counts[node])
            if keep == 0:
                continue
            src = indptr[node]
            dst = new_indptr[node]
            for j in range(keep):
                new_indices[dst + j] = indices[src + j]

        return new_indptr, new_indices

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
    def _expand_pairs_directed_impl(
        values: np.ndarray,
        indptr: np.ndarray,
        kept_nodes: np.ndarray,
        offsets: np.ndarray,
        pairwise: np.ndarray,
        radii: np.ndarray,
        reuse_sources: np.ndarray,
        reuse_targets: np.ndarray,
        reuse_enabled: int,
    ):
        k = kept_nodes.size
        capacity = offsets[-1]
        if reuse_enabled and reuse_sources.size >= capacity:
            sources = reuse_sources[:capacity]
        else:
            sources = np.empty(capacity, dtype=I32)
        if reuse_enabled and reuse_targets.size >= capacity:
            targets = reuse_targets[:capacity]
        else:
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
    def _expand_pairs_directed_capped_impl(
        values: np.ndarray,
        indptr: np.ndarray,
        kept_nodes: np.ndarray,
        offsets: np.ndarray,
        pairwise: np.ndarray,
        radii: np.ndarray,
        degree_cap: int,
        reuse_sources: np.ndarray,
        reuse_targets: np.ndarray,
        reuse_enabled: int,
    ):
        k = kept_nodes.size
        capacity = offsets[-1]
        if reuse_enabled and reuse_sources.size >= capacity:
            sources = reuse_sources[:capacity]
        else:
            sources = np.empty(capacity, dtype=I32)
        if reuse_enabled and reuse_targets.size >= capacity:
            targets = reuse_targets[:capacity]
        else:
            targets = np.empty(capacity, dtype=I32)
        used = np.zeros(k, dtype=I64)
        max_nodes = pairwise.shape[0]
        degree_usage = np.zeros(max_nodes, dtype=I64)
        pruned = I64(0)

        for idx in range(k):
            node = int(kept_nodes[idx])
            s = indptr[node]
            e = indptr[node + 1]
            c = e - s
            if c <= 1:
                continue
            base = offsets[idx]
            write = 0
            for a in range(c - 1):
                pa = int(values[s + a])
                if pa >= max_nodes:
                    continue
                ra = radii[pa]
                for b in range(a + 1, c):
                    pb = int(values[s + b])
                    if pb >= max_nodes:
                        continue
                    rb = radii[pb]
                    bound = ra if ra < rb else rb
                    if pairwise[pa, pb] <= bound:
                        if degree_usage[pa] >= degree_cap or degree_usage[pb] >= degree_cap:
                            pruned += I64(2)
                            continue
                        sources[base + write] = pa
                        targets[base + write] = pb
                        write += 1
                        sources[base + write] = pb
                        targets[base + write] = pa
                        write += 1
                        degree_usage[pa] += 1
                        degree_usage[pb] += 1
            used[idx] = write

        return sources, targets, used, int(pruned)

    def _expand_pairs_directed(
        values: np.ndarray,
        indptr: np.ndarray,
        kept_nodes: np.ndarray,
        offsets: np.ndarray,
        pairwise: np.ndarray,
        radii: np.ndarray,
        degree_cap: int,
        reuse_sources: np.ndarray,
        reuse_targets: np.ndarray,
        reuse_enabled: int,
    ):
        if degree_cap <= 0:
            sources, targets, used = _expand_pairs_directed_impl(
                values,
                indptr,
                kept_nodes,
                offsets,
                pairwise,
                radii,
                reuse_sources,
                reuse_targets,
                reuse_enabled,
            )
            return sources, targets, used, 0
        return _expand_pairs_directed_capped_impl(
            values,
            indptr,
            kept_nodes,
            offsets,
            pairwise,
            radii,
            int(degree_cap),
            reuse_sources,
            reuse_targets,
            reuse_enabled,
        )

    @nb.njit(cache=True)
    def _expand_pairs_chunked_to_csr(
        values: np.ndarray,
        indptr: np.ndarray,
        keep_mask: np.ndarray,
        chunk_ranges: np.ndarray,
        pairwise: np.ndarray,
        radii: np.ndarray,
        batch_size: int,
        degree_cap: int,
        counts_buffer: np.ndarray,
        degree_buffer: np.ndarray,
        reuse_enabled: int,
        indices_buffer: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
        if reuse_enabled and counts_buffer.size >= batch_size:
            counts = counts_buffer[:batch_size]
            counts[:] = 0
        else:
            counts = np.zeros(batch_size, dtype=I64)
        chunk_emitted = 0
        chunk_max_members = 0
        total_pairs = I64(0)
        degree_pruned = I64(0)
        use_degree_cap = degree_cap > 0
        if use_degree_cap:
            if reuse_enabled and degree_buffer.size >= batch_size:
                degree_usage = degree_buffer[:batch_size]
                degree_usage[:] = 0
            else:
                degree_usage = np.zeros(batch_size, dtype=I64)
        else:
            degree_usage = np.empty(0, dtype=I64)

        chunk_count = chunk_ranges.shape[0]
        for chunk_idx in range(chunk_count):
            start = int(chunk_ranges[chunk_idx, 0])
            end = int(chunk_ranges[chunk_idx, 1])
            if end <= start:
                continue
            volume = int(indptr[end] - indptr[start])
            if volume > chunk_max_members:
                chunk_max_members = volume
            chunk_has_edges = False
            for node in range(start, end):
                if not keep_mask[node]:
                    continue
                s = indptr[node]
                e = indptr[node + 1]
                c = e - s
                if c <= 1:
                    continue
                for a in range(c - 1):
                    pa = int(values[s + a])
                    ra = radii[pa]
                    for b in range(a + 1, c):
                        pb = int(values[s + b])
                        rb = radii[pb]
                        bound = ra if ra < rb else rb
                        if pairwise[pa, pb] <= bound:
                            if use_degree_cap and (
                                degree_usage[pa] >= degree_cap or degree_usage[pb] >= degree_cap
                            ):
                                degree_pruned += I64(2)
                                continue
                            counts[pa] += 1
                            counts[pb] += 1
                            if use_degree_cap:
                                degree_usage[pa] += 1
                                degree_usage[pb] += 1
                            total_pairs += I64(2)
                            chunk_has_edges = True
            if chunk_has_edges:
                chunk_emitted += 1

        total_pairs_int = int(total_pairs)
        if total_pairs_int == 0:
            return (
                np.zeros(batch_size + 1, dtype=I64),
                np.empty(0, dtype=I32),
                0,
                chunk_emitted,
                chunk_max_members,
                int(degree_pruned),
            )

        indptr_out = np.empty(batch_size + 1, dtype=I64)
        acc = I64(0)
        indptr_out[0] = 0
        for i in range(batch_size):
            acc += counts[i]
            indptr_out[i + 1] = acc

        if reuse_enabled and indices_buffer.size >= total_pairs_int:
            indices_out = indices_buffer[:total_pairs_int]
        else:
            indices_out = np.empty(total_pairs_int, dtype=I32)
        heads = indptr_out[:-1].copy()

        for chunk_idx in range(chunk_count):
            start = int(chunk_ranges[chunk_idx, 0])
            end = int(chunk_ranges[chunk_idx, 1])
            if end <= start:
                continue
            for node in range(start, end):
                if not keep_mask[node]:
                    continue
                s = indptr[node]
                e = indptr[node + 1]
                c = e - s
                if c <= 1:
                    continue
                for a in range(c - 1):
                    pa = int(values[s + a])
                    ra = radii[pa]
                    for b in range(a + 1, c):
                        pb = int(values[s + b])
                        rb = radii[pb]
                        bound = ra if ra < rb else rb
                        if pairwise[pa, pb] <= bound:
                            pos_a = heads[pa]
                            indices_out[pos_a] = pb
                            heads[pa] = pos_a + 1
                            pos_b = heads[pb]
                            indices_out[pos_b] = pa
                            heads[pb] = pos_b + 1

        return (
            indptr_out,
            indices_out,
            total_pairs_int,
            chunk_emitted,
            chunk_max_members,
            int(degree_pruned),
        )

    @nb.njit(cache=True)
    def _pairs_to_csr(
        sources: np.ndarray,
        targets: np.ndarray,
        offsets: np.ndarray,
        used: np.ndarray,
        batch_size: int,
    ):
        total_used = I64(0)
        counts = np.zeros(batch_size, dtype=I64)
        for node in range(used.size):
            count = used[node]
            total_used += count
            if count == 0:
                continue
            start_in = offsets[node]
            for j in range(count):
                src = int(sources[start_in + j])
                counts[src] += 1

        total_used_int = int(total_used)
        if total_used_int == 0:
            indptr = np.zeros(batch_size + 1, dtype=I64)
            indices = np.empty(0, dtype=I32)
            return indptr, indices, total_used_int

        indptr = np.empty(batch_size + 1, dtype=I64)
        acc = I64(0)
        indptr[0] = 0
        for i in range(batch_size):
            acc += counts[i]
            indptr[i + 1] = acc

        indices = np.empty(total_used_int, dtype=I32)
        heads = indptr[:-1].copy()
        for node in range(used.size):
            count = used[node]
            if count == 0:
                continue
            start_in = offsets[node]
            for j in range(count):
                src = int(sources[start_in + j])
                pos = heads[src]
                indices[pos] = targets[start_in + j]
                heads[src] = pos + 1

        return indptr, indices, total_used_int

    @nb.njit(cache=True, parallel=True)
    def _count_edges_within_radius(indptr: np.ndarray, indices: np.ndarray, radii: np.ndarray, pairwise: np.ndarray) -> np.ndarray:
        n = indptr.size - 1
        out = np.zeros(n, dtype=I64)
        for i in nb.prange(n):
            s = indptr[i]
            e = indptr[i + 1]
            ri = radii[i]
            cnt = 0
            for p in range(s, e):
                j = indices[p]
                bound = ri if ri < radii[j] else radii[j]
                if pairwise[i, j] <= bound:
                    cnt += 1
            out[i] = cnt
        return out

    @nb.njit(cache=True)
    def _prefix_sum64(x: np.ndarray) -> np.ndarray:
        n = x.size
        out = np.empty(n + 1, dtype=I64)
        acc = I64(0)
        out[0] = 0
        for i in range(n):
            acc += x[i]
            out[i + 1] = acc
        return out

    @nb.njit(cache=True)
    def filter_csr_by_radii_from_pairwise(
        indptr: np.ndarray,
        indices: np.ndarray,
        radii: np.ndarray,
        pairwise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = _count_edges_within_radius(indptr, indices, radii, pairwise)
        new_indptr = _prefix_sum64(counts)
        total = int(new_indptr[-1])
        new_indices = np.empty(total, dtype=I32)

        heads = new_indptr[:-1].copy()
        n = indptr.size - 1
        for i in range(n):
            s = indptr[i]
            e = indptr[i + 1]
            ri = radii[i]
            for p in range(s, e):
                j = indices[p]
                bound = ri if ri < radii[j] else radii[j]
                if pairwise[i, j] <= bound:
                    pos = heads[i]
                    new_indices[pos] = j
                    heads[i] = pos + 1

        return new_indptr, new_indices

    def build_conflict_graph_numba_dense(
        scope_indptr: np.ndarray,
        scope_indices: np.ndarray,
        batch_size: int,
        *,
        segment_dedupe: bool = True,
        chunk_target: int = 0,
        chunk_max_segments: int = 0,
        pairwise: np.ndarray | None = None,
        radii: np.ndarray | None = None,
        degree_cap: int = 0,
        scratch_pool: ScopeBuilderArena | None = None,
        pair_merge: bool = False,
    ) -> ScopeAdjacencyResult:
        if scope_indices.size == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                csr_indptr=np.zeros(batch_size + 1, dtype=I64),
                csr_indices=np.empty(0, dtype=I32),
                max_group_size=0,
                total_pairs=0,
                candidate_pairs=0,
                num_groups=0,
                num_unique_groups=0,
                chunk_pairs_cap=0,
                chunk_pairs_before=0,
                chunk_pairs_after=0,
            )

        if pairwise is None or radii is None:
            raise ValueError(
                "pairwise distances and radii arrays are required for the Numba "
                "conflict-graph builder"
            )

        pairwise_arr = np.ascontiguousarray(np.asarray(pairwise, dtype=np.float64))
        radii_arr = np.asarray(radii, dtype=np.float64)
        degree_cap_value = int(degree_cap) if int(degree_cap) > 0 else 0
        reuse_enabled = 1 if scratch_pool is not None else 0

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
                csr_indptr=np.zeros(batch_size + 1, dtype=I64),
                csr_indices=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=0,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
                chunk_count=1,
                chunk_emitted=0,
                chunk_max_members=0,
                chunk_pairs_cap=0,
                chunk_pairs_before=0,
                chunk_pairs_after=0,
                degree_cap=int(degree_cap_value),
                degree_pruned_pairs=0,
            )

        directed_total_pairs = int(total_pairs * 2)
        candidate_pairs = directed_total_pairs
        keep_mask = keep.astype(np.bool_)
        kept_nodes = np.nonzero(keep_mask)[0].astype(I64)
        if kept_nodes.size == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                csr_indptr=np.zeros(batch_size + 1, dtype=I64),
                csr_indices=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=candidate_pairs,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
                chunk_count=1,
                chunk_emitted=0,
                chunk_max_members=0,
                chunk_pairs_cap=0,
                chunk_pairs_before=0,
                chunk_pairs_after=0,
                degree_cap=int(degree_cap_value),
                degree_pruned_pairs=0,
            )

        chunk_ranges_list, chunk_stats = _chunk_ranges_from_indptr(
            indptr_nodes,
            chunk_target,
            chunk_max_segments,
            keep_mask,
            pair_counts=pair_counts,
            pair_merge=pair_merge,
        )
        chunk_pairs_cap = int(chunk_stats.pair_cap)
        chunk_pairs_before = int(chunk_stats.pair_max_before)
        chunk_pairs_after = int(chunk_stats.pair_max_after)
        chunk_pair_merges = int(chunk_stats.pair_merges)
        chunk_count = len(chunk_ranges_list)
        if chunk_count:
            chunk_ranges_arr = np.asarray(chunk_ranges_list, dtype=np.int64).reshape(chunk_count, 2)
        else:
            chunk_ranges_arr = np.zeros((0, 2), dtype=np.int64)
        use_chunks = chunk_target > 0 and chunk_count > 1
        full_volume = int(indptr_nodes[-1] - indptr_nodes[0]) if indptr_nodes.size else 0

        if not use_chunks:
            pair_counts_kept = pair_counts[keep]
            directed_counts = pair_counts_kept * 2
            offsets = _prefix_sum(directed_counts)
            capacity = int(offsets[-1])
            if capacity == 0:
                return ScopeAdjacencyResult(
                    sources=np.empty(0, dtype=I32),
                    targets=np.empty(0, dtype=I32),
                    csr_indptr=np.zeros(batch_size + 1, dtype=I64),
                    csr_indices=np.empty(0, dtype=I32),
                    max_group_size=int(max_group_size),
                    total_pairs=0,
                    candidate_pairs=candidate_pairs,
                    num_groups=num_groups,
                    num_unique_groups=num_unique_groups,
                chunk_count=chunk_count,
                chunk_emitted=0,
                chunk_max_members=full_volume,
                chunk_pairs_cap=chunk_pairs_cap,
                chunk_pairs_before=chunk_pairs_before,
                chunk_pairs_after=chunk_pairs_after,
                degree_cap=int(degree_cap_value),
                degree_pruned_pairs=0,
            )
            if scratch_pool is not None and capacity > 0:
                reuse_sources = scratch_pool.borrow_sources(capacity)
                reuse_targets = scratch_pool.borrow_targets(capacity)
            else:
                reuse_sources = np.empty(0, dtype=I32)
                reuse_targets = np.empty(0, dtype=I32)
            sources, targets, used_counts, degree_pruned = _expand_pairs_directed(
                node_members,
                indptr_nodes,
                kept_nodes,
                offsets,
                pairwise_arr,
                radii_arr,
                degree_cap_value,
                reuse_sources,
                reuse_targets,
                reuse_enabled,
            )
            csr_indptr, csr_indices, actual_pairs = _pairs_to_csr(
                sources,
                targets,
                offsets,
                used_counts,
                batch_size,
            )
            if actual_pairs == 0:
                return ScopeAdjacencyResult(
                    sources=np.empty(0, dtype=I32),
                    targets=np.empty(0, dtype=I32),
                    csr_indptr=np.zeros(batch_size + 1, dtype=I64),
                    csr_indices=np.empty(0, dtype=I32),
                    max_group_size=int(max_group_size),
                    total_pairs=0,
                    candidate_pairs=candidate_pairs,
                    num_groups=num_groups,
                    num_unique_groups=num_unique_groups,
                chunk_count=chunk_count,
                chunk_emitted=0,
                chunk_max_members=full_volume,
                chunk_pairs_cap=chunk_pairs_cap,
                chunk_pairs_before=chunk_pairs_before,
                chunk_pairs_after=chunk_pairs_after,
                chunk_pair_merges=chunk_pair_merges,
                degree_cap=int(degree_cap_value),
                degree_pruned_pairs=0,
            )
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                csr_indptr=csr_indptr,
                csr_indices=csr_indices,
                max_group_size=int(max_group_size),
                total_pairs=actual_pairs,
                candidate_pairs=candidate_pairs,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
                chunk_count=chunk_count,
                chunk_emitted=chunk_count if actual_pairs > 0 else 0,
                chunk_max_members=full_volume,
                chunk_pairs_cap=chunk_pairs_cap,
                chunk_pairs_before=chunk_pairs_before,
                chunk_pairs_after=chunk_pairs_after,
                chunk_pair_merges=chunk_pair_merges,
                degree_cap=int(degree_cap_value),
                degree_pruned_pairs=int(degree_pruned),
            )

        if scratch_pool is not None:
            counts_buffer = scratch_pool.borrow_counts(batch_size)
            degree_buffer = scratch_pool.borrow_degree_usage(batch_size)
            indices_buffer = scratch_pool.borrow_indices(max(1, directed_total_pairs))
        else:
            counts_buffer = np.empty(0, dtype=I64)
            degree_buffer = np.empty(0, dtype=I64)
            indices_buffer = np.empty(0, dtype=I32)

        (
            global_indptr,
            global_indices,
            total_pairs_accum,
            chunk_emitted,
            chunk_max_members,
            degree_pruned_pairs,
        ) = _expand_pairs_chunked_to_csr(
            node_members,
            indptr_nodes,
            keep_mask,
            chunk_ranges_arr,
            pairwise_arr,
            radii_arr,
            batch_size,
            degree_cap_value,
            counts_buffer,
            degree_buffer,
            reuse_enabled,
            indices_buffer,
        )

        if total_pairs_accum == 0:
            return ScopeAdjacencyResult(
                sources=np.empty(0, dtype=I32),
                targets=np.empty(0, dtype=I32),
                csr_indptr=np.zeros(batch_size + 1, dtype=I64),
                csr_indices=np.empty(0, dtype=I32),
                max_group_size=int(max_group_size),
                total_pairs=0,
                candidate_pairs=candidate_pairs,
                num_groups=num_groups,
                num_unique_groups=num_unique_groups,
                chunk_count=chunk_count,
                chunk_emitted=chunk_emitted,
                chunk_max_members=int(chunk_max_members),
                chunk_pairs_cap=chunk_pairs_cap,
                chunk_pairs_before=chunk_pairs_before,
                chunk_pairs_after=chunk_pairs_after,
                chunk_pair_merges=chunk_pair_merges,
                degree_cap=int(degree_cap_value),
                degree_pruned_pairs=int(degree_pruned_pairs),
            )

        return ScopeAdjacencyResult(
            sources=np.empty(0, dtype=I32),
            targets=np.empty(0, dtype=I32),
            csr_indptr=global_indptr,
            csr_indices=global_indices,
            max_group_size=int(max_group_size),
            total_pairs=int(total_pairs_accum),
            candidate_pairs=candidate_pairs,
            num_groups=num_groups,
            num_unique_groups=num_unique_groups,
            chunk_count=chunk_count,
            chunk_emitted=chunk_emitted,
            chunk_max_members=int(chunk_max_members),
            chunk_pairs_cap=chunk_pairs_cap,
            chunk_pairs_before=chunk_pairs_before,
            chunk_pairs_after=chunk_pairs_after,
            chunk_pair_merges=chunk_pair_merges,
            degree_cap=int(degree_cap_value),
            degree_pruned_pairs=int(degree_pruned_pairs),
        )

else:  # pragma: no cover - executed when numba missing

    def _build_scope_csr_from_pairs_impl(
        owners: np.ndarray,
        members: np.ndarray,
        num_nodes: int,
        limit: int,
        top_levels: np.ndarray,
        parents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _require_numba()
        raise AssertionError("unreachable")


def build_scope_csr_from_pairs(
    owners: np.ndarray,
    members: np.ndarray,
    num_nodes: int,
    *,
    limit: int = 0,
    top_levels: np.ndarray | None = None,
    parents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    owners_arr = np.ascontiguousarray(owners, dtype=np.int64)
    members_arr = np.ascontiguousarray(members, dtype=np.int32)
    levels_arr = (
        np.ascontiguousarray(top_levels, dtype=np.int64)
        if top_levels is not None
        else np.empty(0, dtype=np.int64)
    )
    parents_arr = (
        np.ascontiguousarray(parents, dtype=np.int64)
        if parents is not None
        else np.empty(0, dtype=np.int64)
    )
    limit_value = int(limit) if int(limit) > 0 else 0
    return _build_scope_csr_from_pairs_impl(
        owners_arr,
        members_arr,
        int(num_nodes),
        limit_value,
        levels_arr,
        parents_arr,
    )

    def build_conflict_graph_numba_dense(
        scope_indptr: np.ndarray,
        scope_indices: np.ndarray,
        batch_size: int,
        *,
        segment_dedupe: bool = True,
        chunk_target: int = 0,
        chunk_max_segments: int = 0,
        pairwise: np.ndarray | None = None,
        radii: np.ndarray | None = None,
    ) -> ScopeAdjacencyResult:
        _require_numba()
        raise AssertionError("unreachable")

    def filter_csr_by_radii_from_pairwise(
        indptr: np.ndarray,
        indices: np.ndarray,
        radii: np.ndarray,
        pairwise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _require_numba()
        raise AssertionError("unreachable")


if NUMBA_SCOPE_AVAILABLE:
    warmup_scope_builder()


__all__ = [
    "NUMBA_SCOPE_AVAILABLE",
    "build_scope_csr_from_pairs",
    "build_conflict_graph_numba_dense",
    "filter_csr_by_radii_from_pairwise",
    "warmup_scope_builder",
]
