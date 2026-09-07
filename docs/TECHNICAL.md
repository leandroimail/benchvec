# BenchVec — Technical Reference

Complete reference for BenchVec's engines, workloads, configuration,
outputs, statistics, and validated results. For step-by-step usage see
the [User Guide](USER_GUIDE.md); for a quick start see the
[README](../README.md). Portuguese version: [TECHNICAL_pt-BR.md](TECHNICAL_pt-BR.md).

Every claim here is verified against the code in `benchvec/` and the
test suite (`uv run pytest -q` → 305 passed).

---

## 1. What BenchVec measures

BenchVec compares **embedded vector databases** on the same hardware,
dataset, and workload protocol, reporting per engine:

| Metric | Meaning |
|---|---|
| `load_time_s` / `build_time_s` | Ingest wall time / index construction time |
| `mean/p95/p99_latency_ms` | Query latency (serial or per-thread) |
| `qps` | Sustained query throughput |
| `recall_at_k` | Fraction of exact nearest neighbors returned |
| `peak_ram_mb` / `disk_size_mb` | Process peak RAM / persisted footprint |
| `reopen_time_s` | Time to reopen persisted state (B3) |
| `delete_time_s` / `upsert_time_s` / `post_delete_recall` | Lifecycle metrics (D1) |
| `filter_selectivity` | Observed fraction of records matching a filter (B1) |

Every run also records a `host_fingerprint` (hardware/software) and
per-query raw rows (`raw_queries_*.parquet`), so results are auditable
down to individual measurements.

## 2. Engines

All adapters implement the same contract (`benchvec/engines/base.py`):
`connect / reset / insert / build_index / search / search_filtered /
delete / upsert / persist / reopen / disk_size / describe_runtime`.

| Engine | Index | Search params | Metrics | Native filtering | Notes |
|---|---|---|---|---|---|
| **faiss** | `flat` or `ivfflat` (`faiss_nlist`) | `faiss_nprobe` (0 = library default) | l2, cosine | No (app-layer pre-filter) | cosine = normalized vectors + inner product |
| **chromadb** | HNSW (`ef` fixed at collection creation) | `search_ef`; optional rebuild-per-point `ef` grid | l2, cosine | Yes (`where`) | ef grid requires re-creating the collection |
| **lancedb** | `ivfpq` (`num_partitions`, `num_sub_vectors`) | `nprobes`, `refine_factor` | l2, cosine | Yes (SQL `where`) | PQ training needs ≥ 256 rows; fewer rows silently fall back to brute force |
| **sqlite_vec** | `vec0` virtual table (exhaustive KNN) | none (brute force by design) | l2, cosine | No (app-layer pre-filter) | metadata in a companion SQL table (vec0 rejects NULL metadata) |
| **turbovec** | TurboQuant exhaustive scan (`IdMapIndex`), 2/3/4-bit | `bit_width` (build-time; rebuild per sweep point) | **cosine only** | No (app-layer filter → uint64 allowlist pushdown) | Rust/PyO3; kernel is inner-product-only; `dim` must be a multiple of 8; optional TQ+ calibration (~1024-row sample) |

Config-level guards:

- Enabling `turbovec` with `distance_metric = "l2"` fails **at config
  load** (the turbovec kernel cannot evaluate L2).
- The turbovec adapter maps string ids to dense uint64s with
  tombstone-on-delete (slots are never reused or shifted), keeps
  metadata in a JSON sidecar, and retains its calibration sample so
  bit-width sweep rebuilds stay calibrated.

## 3. Workloads

Ten workloads (`benchvec/cli.py::DEFAULT_WORKLOADS`). Each produces one
`BenchmarkSummary` row per engine (per k value / concurrency level /
dataset size where applicable):

| ID | Name | What it does |
|---|---|---|
| `a1` | Ingest | Batched insert; measures `load_time_s`, ingest throughput, RAM |
| `a2` | Search | Exact-recall search at each `k_values` entry (k-sweep: ground truth computed once at max k, sliced per k) |
| `b1` | Filtered search | Metadata-filtered search over 4 deterministic filter scenarios; records `filter_selectivity` |
| `b2` | Rate-limited ingest | Insert + interleaved searches under an ingest-rate cap |
| `b3` | Persistence | `persist()` then measure `reopen_time_s` and post-reopen search |
| `b4` | Reopen | Search against previously persisted state |
| `c1` | Concurrency | Thread ladder (`concurrent_workers`); wall-clock QPS; a failed concurrent query is retried once under a lock and the level is marked `partial` |
| `c2` | Scaling ladder | Fixed query set against growing dataset prefixes (`sample_sizes`); prefixes are chosen by stable hashing so every engine sees identical subsets |
| `d1` | Delete / upsert | Deletes a deterministic fraction (same `:scale` ordering as C2), measures `delete_time_s` and `post_delete_recall` (vs the full pre-delete ground truth), then upserts live-ranked and resurrection records measuring `upsert_time_s` |
| `sweep` | Parameter sweep | Recall-QPS points over engine search-parameter grids (see §5); the sweep runs at `k = k_values[0]` |

Workloads `a1`–`d1` repeat for `workloads.repetitions` rounds; `c2` and
`sweep` run once per engine (their grid points are repetition-insensitive).

## 4. Exact ground truth

`benchvec/ground_truth.py` computes exact nearest neighbors with the
**configured metric** (`engines.distance_metric`):

- `l2`: squared euclidean distance (historical default).
- `cosine`: both matrices are zero-safe L2-normalized, then ranked by
  inner product — identical ordering to cosine distance; the reported
  `distance` is `1 − similarity`.

Metric-aware ground truth matters: the default dataset's embeddings are
**not** unit-norm (norms ≈ 14.9–19.5 measured on
`not-lain/wikipedia-small-3000-embedded`), so scoring cosine-ordered
hits against an L2 ranking would silently bias recall.

Exact engines (sqlite_vec, turbovec 4-bit at low k) scoring
`recall = 1.0` under cosine is the built-in sanity check that the GT
metric matches the search metric.

## 5. Parameter sweep (Recall–QPS frontier)

`run_param_sweep` builds the cross-product of one engine's grid and
measures every point (5 warmup queries + all queries, per-point
latency/recall recorded in raw rows with `search_params`):

| Engine | Grid keys | Application |
|---|---|---|
| faiss | `nprobe` | live (`set_query_params`) |
| lancedb | `nprobes`, `refine_factor` | live (attribute set) |
| chromadb | `ef` (only if `ef_search_grid_enabled`) | rebuild collection + re-ingest per point |
| turbovec | `bit_width` | rebuild quantized index + re-ingest per point (recalibrates from the retained sample) |

`max_points` caps total measured points; over-budget grids are evenly
subsampled so the frontier's high-cost region is not amputated. Engines
without a grid (chromadb by default, sqlite_vec) are skipped. The sweep
runs at `k = k_values[0]` — with the impact profile's
`k_values = [10, 1, 50, 100]` that is k=10, where quantization bit
widths genuinely separate (at k=1 they all saturate at recall 1.0).

## 6. Statistics

`benchvec/statistics.py` (enabled by default via `report --stats`):

- **`stats_summary.parquet`**: per `(workload, engine, k,
  concurrency_level, dataset_size)` group — n, mean latency with a
  seeded bootstrap CI, p95, cross-repetition CV, mean recall.
- **`stats_pairwise.parquet`**: for every engine pair inside a group
  with ≥ `MIN_PAIRWISE_SAMPLES` samples — Mann-Whitney U (asymptotic,
  continuity-corrected), relative mean delta, and a Holm step-down
  correction across the family's pairs. Sweep rows are excluded from
  pairwise tests (heterogeneous grids), and grouping includes
  `dataset_size` so scaling-ladder points are not pooled.

## 7. Charts

`main.py charts` generates ten PNGs (`benchvec/report.py`):

| File | Content |
|---|---|
| `chart_01_latency.png` | mean/p95/p99 latency per engine × workload |
| `chart_02_qps.png` | QPS per engine × workload |
| `chart_03_recall.png` | Recall@K per engine × workload |
| `chart_04_peak_ram.png` | peak RAM per engine × workload |
| `chart_05_disk_footprint.png` | disk footprint per engine |
| `chart_06_build_time.png` | build time / time-to-serve |
| `chart_07_concurrency.png` | QPS per concurrency level (C1) |
| `chart_08_pareto.png` | Recall vs QPS frontier per search-params point (sweep) |
| `chart_09_scaling.png` | mean latency and ingest throughput per dataset size (C2) |
| `chart_10_k_sweep.png` | Recall@K per k (A2 k-sweep) |

`generate_analysis` additionally renders a significance table from
`stats_pairwise.parquet` into the markdown/HTML analysis.

## 8. Configuration

Full schema (`benchvec/config.py`, Pydantic, `extra="forbid"`), TOML or
YAML. Only the most relevant keys; defaults apply when omitted.

| Section | Key | Type | Default | Notes |
|---|---|---|---|---|
| — | `project_name` | str | `benchvec` | |
| — | `seed` | int | 42 | Sampling / ground-truth seed |
| `[paths]` | `data_dir` / `benchmark_dir` / `cache_dir` | path | `.data` / `.benchmarks` / `.cache` | |
| `[dataset]` | `primary` | enum | `fast_wikipedia` | `cohere_wikipedia`, `simple_wikipedia`, `fast_wikipedia`, `sift1m`, `gist1m` |
| | `sample_size` | int | 5000 | Vectors to prepare |
| | `ground_truth_k` / `ground_truth_query_count` | int | 100 / 100 | Exact-GT depth and query count |
| `[engines]` | `enabled` | list | `[faiss, chromadb, lancedb]` | + `sqlite_vec`, `turbovec` |
| | `distance_metric` | enum | `l2` | `turbovec` requires `cosine` (fail-fast at load) |
| | `faiss_index_type` / `faiss_nlist` / `faiss_nprobe` | — | `flat` / 100 / 0 | nprobe 0 = library default |
| `[engines.lancedb]` | `index_type` / `num_partitions` / `num_sub_vectors` / `nprobes` / `refine_factor` | — | `ivfpq` / 8 / 16 / 8 / 1 | refine_factor ≥ 2 enables refinement |
| `[engines.chroma]` | `search_ef` / `ef_search_grid_enabled` / `ef_search_grid` | — | 64 / false / [] | grid rebuilds per point |
| `[engines.turbovec]` | `bit_width` / `calibrate` | 2·3·4 / bool | 4 / false | build-time knob |
| `[workloads]` | `repetitions` / `warmup_queries` / `k_values` / `ingest_batch_sizes` / `ingest_rates_per_second` / `concurrent_workers` / `sample_sizes` / `max_c2_queries` | — | 3 / 10 / [10] / [1000] / [100] / [1] / [] / 20000 | `k_values[0]` drives the sweep |
| `[workloads.param_sweep]` | `faiss_nprobe` / `lancedb_nprobes` / `lancedb_refine_factor` / `turbovec_bit_width` / `max_points` | list[int] / int | [] / [] / [] / [] / 20000 | grids; `turbovec_bit_width ∈ {2,3,4}` |
| `[timeouts]` | per-phase seconds | int | 1800/1800/300/900/600 | partial results captured on timeout |

## 9. CLI

`uv run python main.py <command>` (entrypoint `main.py`):

| Command | Purpose |
|---|---|
| `run` | Selected workloads (`--workload` repeatable) for enabled engines; `--config`, `--project` (default `benchmark_default`), `--output-dir`, `--keep-engine-storage` |
| `run-all` | The full default workload suite (all 10 workloads) |
| `report` | Consolidate all summaries into one parquet (`--input`/`--project`, `--output`; `--stats/--no-stats`, default on) |
| `charts` | Generate the 10 PNG charts (`--input`, `--output`) |
| `analyze` | Markdown/HTML/JSON analysis (`--format`, `--report-type`) |
| `inspect` | Inspect one run summary or its raw records (`--project`, `--run-id`, `--summary`, `--input`) |
| `init` | Create a project folder (`--project`, `--template <config.toml>`) |
| `download` | Download and cache the configured dataset |

Useful flags:

- `--keep-engine-storage` retains per-run engine storage (default:
  delete after metrics are captured — disk hygiene; shared C1 storage
  is cleaned by the CLI after the invocation).
- Always pass `--config configs/impact-benchmark.toml` explicitly for
  the full profile.

## 10. Projects and outputs

```
.benchmarks/projects/<project>/
├── project.toml            # project metadata
├── config.toml             # optional config copy (init --template)
├── results/<engine>/<workload>/<run_id>/
│   ├── summary.parquet     # one BenchmarkSummary row
│   ├── env.json            # host fingerprint + library versions
│   └── raw_queries_*.parquet / raw_ingest_*.parquet
├── reports/                # consolidated parquet, stats parquets, charts/, analysis
└── logs/
```

`benchmark_default` is the default project when `--project` is omitted.
Note: `.benchmarks/` is gitignored — artifacts are reproducible, not
archived.

## 11. Pre-built profiles

| Config | Purpose |
|---|---|
| `configs/first-benchmark.toml` | Official full benchmark (Cohere Wikipedia) |
| `configs/fast-benchmark.toml` | Fast smoke profile |
| `configs/impact-benchmark.toml` | **Full 10-workload campaign, 5 engines, cosine** — the headline profile |

Impact profile, measured on a MacBook M5 (16 GB, warm dataset cache):

- 233 summaries, **all `status=success`**; ~8.8 min workload wall time
  (clean run); 9.4 MB on disk with default engine-storage cleanup.
- Engines: faiss (IVFFlat nprobe 1–32), lancedb (IVF_PQ nprobes
  1–16), chromadb (single ef point), sqlite_vec (no grid), turbovec
  (bit-width 2/3/4 rebuild grid).
- TurboVec bit-width sweep at k=10: recall 0.899 / 0.956 / 0.973 at
  ~0.09 ms — a genuine Pareto spread (at k=1 all widths saturate at
  1.0, hence `k_values = [10, 1, 50, 100]`).
- TurboVec C1 scales natively (≈3.3–4.5k QPS at 1 thread → 6.3–7.4k at
  8 threads; no locked-retry `partial` results).
- Measured quirk (turbovec 1.0.0): 3-bit and 4-bit index snapshots are
  the same size (internal 4-bit packing); only 2-bit compresses
  (~0.53× of 4-bit).
- Sanity anchor: sqlite_vec (exact brute force) scores recall 1.0 at
  every k under cosine, confirming ground-truth/search metric
  alignment across the campaign.

## 12. Design decisions and limitations

- **Metric-aware ground truth**: GT honors `engines.distance_metric`;
  the impact profile uses cosine because turbovec is inner-product-only
  and engines are only comparable within one metric.
- **TurboVec id mapping**: dense uint64 assignment with
  tombstone-on-delete; `ids.json`/`metadata.json` sidecars; a snapshot
  without a usable ids sidecar is discarded on reopen (fresh state)
  rather than diverging.
- **TurboVec documents are not stored** (hits carry id + score only);
  metadata is kept solely for filter evaluation. Metrics never read
  hit payloads beyond ids.
- **chroma `ef` and turbovec `bit_width` are build-time knobs**: sweep
  points rebuild the index and re-ingest (measured inside the sweep).
- **faiss deletes are tombstones**: space is not reclaimed until a
  rebuild; D1 measures the behavior as-is.
- **lancedb PQ training floor**: < 256 rows silently falls back to
  brute force; the adapter warns and skips IVF_PQ at tiny ladder sizes.
- **C1 concurrency**: engines without thread-safe clients surface
  honest `partial` levels via a locked-retry path (turbovec, faiss,
  chroma, lancedb and sqlite_vec all run native in practice).
- **Timeouts** capture partial results per phase (`benchvec/timeouts.py`)
  instead of failing the run.

## 13. Testing

```bash
uv run pytest -q          # 305 passed
```

Covers config validation, every engine adapter (including turbovec:
round-trips, tombstones, filters, persistence, calibration, rebuild,
concurrency), ground-truth metrics, runner orchestration, CLI commands,
reporting/statistics, and the analysis generator. Tests skip cleanly
when an optional native dependency is unavailable on the host.
