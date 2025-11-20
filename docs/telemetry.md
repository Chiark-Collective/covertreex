# Telemetry Schema & Renderer

`BenchmarkLogWriter` emits a JSONL row for every insertion batch (or prefix group)
processed by the CLI. Each row conforms to the
`covertreex.benchmark_batch.v2` schema:

- `schema_id`, `schema_version` — identify the payload version. Future tools can
  reject incompatible logs up front.
- `run_id`, `run_hash` — `run_id` is human-readable while `run_hash` is a
  deterministic SHA‑256 digest of the runtime configuration. The digest is
  derived from the `runtime_digest` (a hash of the fully materialised
  `RuntimeModel`) plus the `seed_pack`. Two runs with identical configs and
  seeds therefore share a hash and can be diffed quickly.
- `runtime_digest`, `seed_pack` — `runtime_digest` is the canonical config hash
  used when computing `run_hash`, while `seed_pack` lifts the effective channel
  seeds (global/mis/batch/residual grid) to the header without trawling the
  nested runtime snapshot.
- `runtime` — full `RuntimeModel` snapshot (backend, metric, diagnostics,
  residual policy, seeds). Flattened `runtime_*` keys remain for backwards
  compatibility.
- `metadata` — benchmark-provided annotations such as `benchmark` name or tree
  dimensions.
- Measurement fields — every traversal/conflict/gate counter surfaces as a
  top-level numeric field on the same row. Residual batches now write
  `residual_batch_whitened_pair_share`, `residual_batch_whitened_time_share`,
  and ratio companions so downstream tooling can compute coverage summaries
  without reprocessing footage.

Example JSONL record (trimmed for brevity):

```json
{
  "schema_id": "covertreex.benchmark_batch.v2",
  "schema_version": 2,
  "run_id": "pcct-20250106-120010-a1b2c3",
  "run_hash": "f58f6c9a...",
  "runtime_digest": "3c8d20f1...",
  "batch_index": 0,
  "batch_size": 512,
  "runtime": {
    "backend": "numpy",
    "metric": "residual",
    "seeds": {"mis": 0, "batch_order": 0, "global_seed": null, "residual_grid": null}
  },
  "seed_pack": {"global_seed": null, "mis": 0, "batch_order": 0, "residual_grid": null},
  "metadata": {"benchmark": "pcct.query", "dimension": 8},
  "traversal_ms": 42.1,
  "conflict_graph_ms": 18.4,
  "mis_ms": 3.9,
  "residual_batch_whitened_pair_share": 0.82,
  "residual_batch_whitened_time_share": 0.77,
  "conflict_pairwise_reused": 1
}
```

## `pcct telemetry render`

Use the new Typer subcommand to summarise logs without spreadsheets:

```bash
# JSON summary (default)
python -m cli.pcct telemetry render artifacts/benchmarks/queries_run.jsonl

# Markdown table with the list of measurement fields
python -m cli.pcct telemetry render artifacts/benchmarks/queries_run.jsonl \
  --format md --show fields

# key/value CSV for quick ingestion into dashboards
python -m cli.pcct telemetry render artifacts/benchmarks/queries_run.jsonl --format csv > summary.csv
```

Sample Markdown output:

```
## Run Summary
- schema: covertreex.benchmark_batch.v2 (v2)
- run: pcct-20250106-120010-a1b2c3 (hash=f58f6c9a...)
- batches: 8 (points=4096)
- runtime: backend=numpy, metric=residual, enable_numba=True

### Measurements
| metric | mean | p50 | p90 | p95 | max | samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| traversal_ms | 41.222 | 40.981 | 42.903 | 43.115 | 44.022 | 8 |
| conflict_graph_ms | 17.884 | 17.502 | 18.990 | 19.225 | 19.884 | 8 |
| mis_ms | 3.912 | 3.850 | 4.120 | 4.201 | 4.250 | 8 |

### Residual coverage
- whitened pairs: 65634 (82.0% of pairs)
- whitened time: 15.20 ms (78.5% of time)

### Gate metrics
(Deprecated/Removed in v2)

# Measurement fields
traversal_ms
traversal_whitened_block_pairs
...
```

CSV summaries follow the `key,value` shape so they slot into basic dashboards
without custom parsers. JSON output mirrors the internal summary dictionary and
is suitable for scripting when richer automation is needed.

## Residual annotations

Residual traversal batches now emit per-record ratios in addition to the raw
`whitened_block_*` counters:

- `residual_batch_whitened_pair_share` — whitened/(whitened + kernel) pairs.
- `residual_batch_whitened_time_share` — whitened/(whitened + kernel) time.
- `residual_batch_whitened_pair_ratio` and `...time_ratio` — guard-rail ratios
  (∞ denotes kernel tiles absent).

`pcct telemetry render` aggregates these fields to replicate the CLI summary
that previously printed to stdout, making it easier to archive evidence for
auditors.
