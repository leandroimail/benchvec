# BenchVec User Guide — Complete Reference

> **BenchVec** is a Python benchmarking suite for comparing embedded vector databases (FAISS, ChromaDB, LanceDB, sqlite-vec, TurboVec) with reproducibility and scientific rigor.

---

## Table of Contents

1. [Quick Start ](#quick-start)
2. [Detailed Installation](#installation)
3. [Official Datasets](#official-datasets)
4. [Project System](#project-system)
5. [Benchmark Configuration](#configuration)
6. [Running Benchmarks](#running-benchmarks)
7. [Generating Charts](#generating-charts)
8. [Inspecting Results](#inspecting-results)
9. [Command Reference](#command-reference)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install 

```bash
cd benchvec
uv venv --python 3.12 .venv
uv sync --dev
```

### 2. Validate (30 seconds)

```bash
uv run pytest -q
# Expected: 305 passed
```

### 3. First Benchmark 

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
# Captures run_id in output (e.g: run-3e831116cfa4)
```

### 3b. Fast Test Profile (<5 minutes)

For quick validation without HuggingFace authentication:
```bash
uv run python main.py run-all --config configs/fast-benchmark.toml
```
**Dataset:** `not-lain/wikipedia-small-3000-embedded` — 3,000 records with pre-computed embeddings (public, no HF_TOKEN).

**Available templates:**
- `configs/fast-benchmark.toml` — Fast profile (5K entries, batch=1000, no HF_TOKEN)
- `configs/first-benchmark.toml` — Full profile (requires HF_TOKEN for Cohere Wikipedia)

### 4. View Results (instant)

```bash
RUNID="run-3e831116cfa4"  # your run_id
uv run python main.py inspect --run-id $RUNID --summary
```

**Expected output:** Table with A1-B4 metrics (latency, recall, QPS) by engine.

---

## Main Benchmark

Steps to execute the complete benchmark campaign (all workloads, all engines).

### Prerequisites

1. **Authenticate HuggingFace** (Cohere Wikipedia)
```bash
huggingface-cli login
# or
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

2. **Prepare environment** (if not done)
```bash
uv sync --dev
uv run pytest -q  # Validate: 305 passed
```

### Step 1: Create Project

```bash
uv run python main.py init --project full-benchmark
# Output: ✓ Project initialized: /path/to/.benchmarks/projects/full-benchmark
```

### Step 2: Execute Full Benchmark

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
```

**What will be tested:**
- **Engines**: FAISS (IndexFlatL2), ChromaDB (HNSW), LanceDB (Lance)
- **Dataset**: Cohere Wikipedia 22-12 (180 passages, 1024 dimensions)
- **Workloads**: A1-B4 (ingest, search, filtered, persistence, reopen, RAM pressure); C1 (concurrent search ladder); C2 (dataset scaling ladder); D1 (delete/upsert lifecycle); sweep (Recall-QPS parameter sweep); k-sweep in A2
- **Repetitions**: 10 reps × 3 engines = 30 runs

### Step 3: Consolidate Results

```bash
uv run python main.py report \
  --input .benchmarks/projects/full-benchmark/results \
  --output .benchmarks/projects/full-benchmark/reports/final.parquet
```

Export to CSV (Excel compatible):

```bash
uv run python main.py report \
  --input .benchmarks/projects/full-benchmark/results \
  --output .benchmarks/projects/full-benchmark/reports/final.csv
```

### Step 4: Generate Charts

```bash
uv run python main.py charts --project full-benchmark
# Output: Generated 10 charts in .benchmarks/projects/full-benchmark/reports/charts/
```

**Output**: 10 PNGs in `.benchmarks/projects/full-benchmark/reports/charts/`

### Step 5: Prepare Documentation

Record environment fingerprint for the paper:

```bash
# View host fingerprint (CPU, RAM, storage, OS)
cat .benchmarks/projects/full-benchmark/results/faiss/a1/run-*/env.json | jq '.host_fingerprint'

# View library versions
cat .benchmarks/projects/full-benchmark/results/**/env.json | jq '.library_versions'
```

---

## Installation

### Requirements

- **Python:** 3.10+
- **OS:** Linux, macOS, Windows (WSL2)
- **RAM:** 4+ GB
- **Disk:** 10+ GB (for prepared datasets)

### Step 1: Prepare Environment

```bash
# Setup Python 3.12
uv python install 3.12

# Project folder
cd benchvec
uv venv --python 3.12 .venv

# Dependencies (dev + production)
uv sync --dev
```

### Step 2: Validate

```bash
# Linting
uv run ruff check .
# Expected: 0 errors

# Type checking
uv run mypy benchvec tests
# Expected: Success

# Tests
uv run pytest -q
# Expected: 305 passed
```

### Step 3: Datasets (Optional)

For **Cohere Wikipedia** (recommended):

```bash
# Authenticate (requires HF account)
huggingface-cli login
# Paste token when prompted
```

For **offline**, prepare cache in `.cache/cohere_wikipedia/`:

```
.cache/
└─ cohere_wikipedia/
   ├─ records.parquet
   ├─ selected_ids.json
   ├─ doc_chunk_map.json
   └─ manifest.json
```

---

## Official Datasets

BenchVec supports multiple datasets organized in two scientific groups:

### Group A: Classic ANN Benchmark

Focus on pure vector search efficiency, without structured filters.

#### SIFT1M (Recommended for Papers)
- **Dimension:** 128
- **Size:** 1 million vectors
- **Queries:** 10,000
- **Ground truth:** Available
- **Usage:** Canonical benchmark in vector search, allows comparison with related work
- **URL:** http://sift.fi.msu.ru/
- **Expected performance:** FAISS ~0.016ms (Flat), ChromaDB ~1-2ms, LanceDB ~0.5ms

#### GIST1M
- **Dimension:** 960
- **Size:** 1 million vectors
- **Queries:** 1,000 (note: many publications use only 1k!)
- **Ground truth:** Available
- **Note:** High dimension; p95/p99 less significant (n=1000)

### Group B: Local RAG Workload with Metadata

Focus on realistic scenarios with persistence, structured filters, and metadata.

#### Cohere Wikipedia 22-12 (Main - Recommended)
- **Corpus:** Simple English Wikipedia with pre-computed embeddings
- **Dimension:** 1024 (model `embed-multilingual-v2.0`)
- **Size:** Up to 170,000 passages (use configurable sample_size)
- **Prerequisite:** HuggingFace authentication (CLI login)
- **Advantage:** Pre-computed embeddings = guaranteed reproducibility
- **Configuration:**
  ```toml
  [dataset]
  primary = "cohere_wikipedia"
  sample_size = 100000  # or 10000 for quick test
  ```

#### Wikipedia 20220301 (Public Alternative)
- **Corpus:** Simple English Wikipedia raw text (no pre-computed embeddings)
- **Dimension:** 384 (model `sentence-transformers/all-MiniLM-L6-v2`)
- **Size:** Adjustable; ~200k chunks after chunking
- **Advantage:** No HuggingFace authentication required
- **Pipeline:** Text → Chunking (TokenChunker, 256 tokens) → Local embedding
- **Configuration:**
  ```toml
  [dataset]
  primary = "simple_wikipedia"
  chunk_size = 256
  chunk_overlap = 32
  ```

#### Structured Metadata (Synthetic)
All Group B datasets include metadata for filter testing:
- **category**: derived by keywords (data_engineering, rag, general, etc)
- **tag**: filter-controlled tags per category
- **author**: deterministic distribution (8 values)
- **creation_date**: deterministic synthetic timestamps (2021-2025)
- **document_length_bucket**: size ranges (short/medium/long/xlong)

### Recommendation for Publications

1. **For Benchmark A (Pure ANN):** Use **SIFT1M** (canonical dataset, won't be questioned)
2. **For Benchmark B (Local RAG):** Use **Cohere Wikipedia**
3. **Default Configuration:**
   ```toml
   [dataset]
   primary = "cohere_wikipedia"
   sample_size = 100000
   ground_truth_k = 100
   ```

---

## Project System

BenchVec organizes each benchmark in an independent **project** with standardized structure.

### What is a Project?

A project is a directory that groups:
- **Configuration** of benchmark (project.toml)
- **Results** of all executions
- **Consolidated** reports
- **Charts** PNG for comparison
- **Logs** of execution

### Project Structure

```
.benchmarks/projects/my-project/
├── project.toml              # Project metadata
├── config.toml               # Default config (optional)
├── results/                  # Execution results
│   ├── faiss/
│   │   ├── a1/run-xxx/
│   │   ├── a2/run-yyy/
│   │   └── ...
│   ├── chromadb/
│   ├── lancedb/
│   └── index.parquet
├── reports/                  # Consolidated reports
│   ├── final.parquet
│   ├── stats_summary.parquet
│   ├── stats_pairwise.parquet
│   └── charts/               # 10 PNG charts
│       ├── chart_01_latency.png
│       ├── chart_02_qps.png
│       ├── chart_03_recall.png
│       ├── chart_04_peak_ram.png
│       ├── chart_05_disk_footprint.png
│       ├── chart_06_build_time.png
│       ├── chart_07_concurrency.png
│       ├── chart_08_pareto.png
│       ├── chart_09_scaling.png
│       └── chart_10_k_sweep.png
└── logs/                     # Execution logs
```

### Create a New Project

```bash
# Create standard structure
uv run python main.py init --project my-benchmark

# Expected output:
# ✓ Project initialized: /path/to/.benchmarks/projects/my-benchmark
# Results: /path/to/.benchmarks/projects/my-benchmark/results
# Reports: /path/to/.benchmarks/projects/my-benchmark/reports
# Logs: /path/to/.benchmarks/projects/my-benchmark/logs
```

### Use a Project

```bash
# Run benchmark in specific project
uv run python main.py run-all \
  --config configs/first-benchmark.toml \
  --project my-benchmark

# Generate project charts
uv run python main.py charts --project my-benchmark

# View results
open .benchmarks/projects/my-benchmark/reports/charts/
```

### Default Projects

- **benchmark_default**: Default project when none specified
- Created automatically on first execution

---

## Project Structure

The project root and benchmark projects follow this hierarchy:

```
benchvec/
├── README.md                     # Main documentation (English)
├── README_pt-BR.md              # Main documentation (Portuguese)
├── LICENSE                       # MIT License
├── benchvec/                    # Main package
│   ├── __init__.py
│   ├── cli.py                   # CLI commands
│   ├── config.py                # Configuration
│   ├── datasets.py              # Dataset loading
│   ├── chunker.py               # Text segmentation
│   ├── ground_truth.py          # Ground truth
│   ├── metrics.py               # Measurements
│   ├── runner.py                # Benchmark orchestration
│   ├── raw_store.py             # Micro-data storage
│   ├── report.py                # Reports & charts
│   ├── host.py                  # Hardware fingerprinting
│   ├── project.py               # Project management
│   ├── timeouts.py              # Timeout policies
│   └── engines/                 # Engine adapters
│       ├── __init__.py
│       ├── base.py              # Base engine class
│       ├── faiss_engine.py      # FAISS adapter
│       ├── chroma_engine.py   # ChromaDB adapter
│       ├── lancedb_engine.py    # LanceDB adapter
│       ├── sqlite_vec_engine.py # sqlite-vec adapter (brute force)
│       └── turbovec_engine.py   # TurboVec adapter (TurboQuant, cosine-only)
├── configs/                     # Configuration templates
│   └── first-benchmark.toml     # Official benchmark config
├── docs/
│   ├── USER_GUIDE.md           # This file (English guide)
│   ├── COMO_USAR.md            # Portuguese guide
│   ├── TECHNICAL.md            # Complete technical reference (English)
│   └── TECHNICAL_pt-BR.md      # Complete technical reference (Portuguese)
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest configuration
│   └── unit/                    # 305 unit tests
│       ├── test_cli.py
│       ├── test_config.py
│       ├── test_datasets.py
│       ├── test_*.py
│       └── ...
├── .benchmarks/                # Results directory (auto-created)
│   └── projects/
│       ├── benchmark_default/
│       │   ├── project.toml
│       │   ├── results/         # Run results
│       │   │   ├── faiss/
│       │   │   │   ├── a1/run-xxx/
│       │   │   │   │   ├── summary.parquet
│       │   │   │   │   ├── env.json
│       │   │   │   │   └── raw_*.parquet
│       │   │   │   ├── a2/run-yyy/
│       │   │   │   └── ...
│       │   │   ├── chromadb/
│       │   │   ├── lancedb/
│       │   │   └── index.parquet
│       │   ├── reports/         # Consolidated reports
│       │   │   ├── final.parquet
│       │   │   ├── final.csv
│       │   │   └── charts/      # 10 PNG comparison charts
│       │   │       ├── chart_01_latency.png
│       │   │       ├── chart_02_qps.png
│       │   │       ├── chart_03_recall.png
│       │   │       ├── chart_04_peak_ram.png
│       │   │       ├── chart_05_disk_footprint.png
│       │   │       ├── chart_06_build_time.png
│       │   │       ├── chart_07_concurrency.png
│       │   │       ├── chart_08_pareto.png
│       │   │       ├── chart_09_scaling.png
│       │   │       └── chart_10_k_sweep.png
│       │   └── logs/            # Execution logs
│       └── my-project/          # Custom project
│           └── [same structure]
├── .cache/                      # Dataset cache
│   ├── cohere_wikipedia/
│   ├── simple_wikipedia/
│   └── ...
└── .venv/                       # Python virtual environment
```

---

## Configuration

### TOML File

Benchmarks configured via TOML (e.g: [configs/first-benchmark.toml](../configs/first-benchmark.toml)).

#### Minimal Example

```toml
project_name = "my_benchmark"
seed = 42

[paths]
benchmark_dir = ".benchmarks"
cache_dir = ".cache"

[dataset]
primary = "cohere_wikipedia"
sample_size = 180

[engines]
enabled = ["faiss", "chromadb", "lancedb"]
distance_metric = "l2"

[workloads]
repetitions = 10
k_values = [10]
ingest_batch_sizes = [60]
```

#### Key Fields

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-----------|
| — | `project_name` | string | benchvec | Project name |
| — | `seed` | int | 42 | Reproducibility seed |
| **[paths]** | `benchmark_dir` | path | .benchmarks | Results |
| | `cache_dir` | path | .cache | Dataset cache |
| **[dataset]** | `primary` | enum | cohere_wikipedia | Dataset: cohere_wikipedia, simple_wikipedia, sift1m, gist1m |
| | `sample_size` | int | 100_000 | Number of vectors |
| | `ground_truth_k` | int | 100 | K for ground-truth |
| **[engines]** | `enabled` | list | [faiss, chromadb, lancedb] | Engines: faiss, chromadb, lancedb, sqlite_vec, turbovec |
| | `distance_metric` | enum | l2 | Metric: l2 or cosine (turbovec requires cosine — its kernel is inner-product-only; enabling turbovec under l2 fails at config load) |
| | `faiss_index_type` | enum | flat | FAISS: flat or ivfflat |
| **[engines.turbovec]** | `bit_width` | 2\|3\|4 | 4 | TurboQuant bit width (build-time; sweep via `turbovec_bit_width`) |
| | `calibrate` | bool | false | TQ+ calibration (~1024-row sample) at build time |
| **[workloads]** | `repetitions` | int | 10 | Reps per engine |
| | `k_values` | list | [10] | K values for recall@K |
| | `ingest_batch_sizes` | list | [1000] | Batch sizes |

### Pre-built Configs

**Quick Test:** 
```toml
[dataset]
sample_size = 50
[engines]
enabled = ["faiss", "lancedb"]
[workloads]
repetitions = 2
```

**Full Suite:** Use `configs/first-benchmark.toml`

---

## Running Benchmarks

### run-all

Runs all workloads (A1-B4) with all engines:

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
```

**Output:**
```
.benchmarks/projects/benchmark_default/results/
├─ faiss/a1/run-238f968acf79/
│  ├─ summary.parquet       # Metrics
│  ├─ env.json              # Host info
│  └─ raw_ingest_*.parquet  # Micro-data
├─ faiss/a2/run-.../
├─ faiss/b1-b4/.../
├─ chromadb/...
└─ lancedb/...
```

### run

Selected workloads:

```bash
# Only A1 and A2
uv run python main.py run \
  --config configs/first-benchmark.toml \
  --workload a1 --workload a2

# Only FAISS
uv run python main.py run \
  --config configs/first-benchmark.toml \
  --engine faiss
```

---

## Generating Charts

After running a benchmark, automatically generate 10 comparison PNG charts.

### Basic Command

```bash
# Generate all charts
uv run python main.py charts --project my-benchmark
```

### Output

Charts are saved in `.benchmarks/projects/my-benchmark/reports/charts/`:

```
chart_01_latency.png            # Latency mean/p95/p99 by engine
chart_02_qps.png                # Throughput (queries per second)
chart_03_recall.png             # Recall@K by engine (target: 1.0)
chart_04_peak_ram.png           # Peak RAM (MB) by workload
chart_05_disk_footprint.png     # Aggregated disk footprint
chart_06_build_time.png         # Build time + time-to-serve
chart_07_concurrency.png        # QPS per thread level (C1 ladder)
chart_08_pareto.png             # Recall vs QPS frontier per sweep point
chart_09_scaling.png            # Latency/throughput per dataset size (C2)
chart_10_k_sweep.png            # Recall@K per k value (A2 k-sweep)
```

**Total size:** ~240-250 KB (40-50 KB each)

### Customize Output

```bash
# Specify input directory
uv run python main.py charts \
  --project my-benchmark \
  --input .benchmarks/projects/my-benchmark/results

# Specify output directory
uv run python main.py charts \
  --project my-benchmark \
  --output .benchmarks/projects/my-benchmark/reports/v2_charts/
```

### Interpret the Charts

#### Chart 1: Latency
- **Y-axis:** Latency in milliseconds (log recommended)
- **Bars:** Mean with confidence intervals (95%)
- **Colors:** FAISS (blue), ChromaDB (orange), LanceDB (green)
- **Expected:** FAISS ~0.01-0.05ms, ChromaDB/LanceDB ~1-10ms

#### Chart 2: QPS (Queries Per Second)
- **Y-axis:** Throughput in queries/second
- **Scales:** Can differ 10-100x between engines
- **Expected:** FAISS ~50k+, ChromaDB ~500-1k, LanceDB ~600-750

#### Chart 3: Recall@K
- **Y-axis:** Recall (0.0 to 1.0)
- **Expected:** 1.0 on all engines (all find correct top-K)
- **Alert:** If < 1.0, check ground truth or bugs

#### Chart 4: Peak RAM
- **Y-axis:** RAM in MB
- **Workloads:** A1 (ingest) vs A2-B4 (queries)
- **Expected:** 300-500 MB for 100k dataset

#### Chart 5: Disk Footprint
- **Y-axis:** Size in MB (aggregated)
- **Comparison:** ChromaDB > FAISS > LanceDB
- **Expected:** LanceDB ~0.05-0.2 MB, ChromaDB ~1-3 MB

#### Chart 6: Build Time + Time-to-Serve
- **Y-axis:** Time in seconds
- **Phases:** Build time (blue) vs Time-to-serve (orange)
- **Expected:** FAISS ~0.0-0.01s, ChromaDB ~0.02-0.05s

#### Chart 7: Concurrency (C1)
- **Y-axis:** QPS per concurrency level
- **Reading:** linear scaling means the engine uses threads natively; a
  flat line means the client serializes queries
- **Alert:** `status=partial` levels indicate a locked-retry fallback
  (the engine is not thread-safe)

#### Chart 8: Pareto Frontier (Sweep)
- **Axes:** QPS (log) vs mean Recall@K, one line per engine
- **Reading:** each point is one search-parameter grid value (e.g.
  FAISS `nprobe`, LanceDB `nprobes`/`refine_factor`, TurboVec bit width)
- **Expected:** the upper-left frontier dominates; dominated points are
  drawn dimmer

#### Chart 9: Scaling (C2)
- **Y-axis:** Mean latency and ingest throughput per dataset size
- **Reading:** super-linear latency growth signals memory pressure or
  cache exhaustion at larger sizes

#### Chart 10: k-Sweep (A2)
- **Y-axis:** Recall@K for each configured `k_values` entry
- **Expected:** recall decays slowly as k grows; sharp drops indicate
  index quality limits (e.g. aggressive IVF pruning)

### Export for Presentation

```bash
# Copy charts to presentation folder
cp .benchmarks/projects/my-benchmark/reports/charts/*.png ~/presentation/

# In LaTeX/Markdown
![Latency comparison](./charts/chart_01_latency.png)
```

---

## Quick Analysis

Generate a structured analysis report from consolidated benchmark
summaries (Markdown, HTML or JSON):

### Basic Usage

```bash
# Detailed markdown report (default)
uv run python main.py analyze --project my-benchmark --output report.md
```

### Output Formats

```bash
# JSON for programmatic processing
uv run python main.py analyze --project my-benchmark --format json --output analysis.json

# Self-contained HTML, browser-viewable
uv run python main.py analyze --project my-benchmark --format html --output report.html
```

### Report Types

```bash
# High-level overview for presentations
uv run python main.py analyze --project my-benchmark --report-type executive_summary --output summary.md

# Full analysis with comprehensive tables (default)
uv run python main.py analyze --project my-benchmark --report-type detailed --output detailed.md

# Metrics and statistics only
uv run python main.py analyze --project my-benchmark --report-type statistical_only --output stats.md
```

The analysis renders the significance table from
`stats_pairwise.parquet` (run `report --stats` first, default on) and a
workload-by-workload summary. See [TECHNICAL.md](TECHNICAL.md) §6–§7 for
methodology.

---

## Inspecting Results

### inspect

Details of a run:

```bash
# Aggregated summary
uv run python main.py inspect --run-id run-3e831116cfa4 --summary

# Individual queries
uv run python main.py inspect --run-id run-3e831116cfa4 --type queries

# Ingest
uv run python main.py inspect --run-id run-3e831116cfa4 --type ingest
```

### report

Consolidate into a table:

```bash
# Parquet (analysis in Python)
uv run python main.py report \
  --input .benchmarks/projects/benchmark_default/results \
  --output reports/consolidated.parquet

# CSV (Excel)
uv run python main.py report \
  --input .benchmarks/projects/benchmark_default/results \
  --output reports/consolidated.csv
```

### Read in Python

```python
import pandas as pd

# Consolidated table
df = pd.read_parquet('reports/consolidated.parquet')

# Filter
faiss_a2 = df[(df['engine'] == 'faiss') & (df['workload'] == 'a2')]
print(faiss_a2[['mean_latency_ms', 'qps', 'recall_at_k']])

# Raw micro-data
queries = pd.read_parquet('.benchmarks/.../raw_queries_run-XXX.parquet')
```

---

## Complete Command Reference

```bash
# Run-all (standard suite)
uv run python main.py run-all --config configs/first-benchmark.toml

# Run selected
uv run python main.py run --config CONFIG.toml --workload a1 --engine faiss

# Consolidated report
uv run python main.py report --input .benchmarks/.../results --output reports/final.parquet

# Inspect
uv run python main.py inspect --run-id run-XXX --type summary|queries|ingest

# Init (new project)
uv run python main.py init --project my-project
```

---

## Output Structure

```
.benchmarks/projects/benchmark_default/results/
├─ faiss/
│  ├─ a1/run-238f968acf79/
│  │  ├─ summary.parquet           # Metrics: latency, QPS, recall
│  │  ├─ env.json                  # CPU, RAM, storage, Python version
│  │  └─ raw_ingest_*.parquet      # Individual ingest entries
│  ├─ a2/run-f2a82d84b734/
│  │  ├─ summary.parquet
│  │  ├─ env.json
│  │  └─ raw_queries_*.parquet     # Individual query entries
│  ├─ b1-b4/...
├─ chromadb/...
├─ lancedb/...
└─ index.parquet                    # Index: run_id → summary_path
```

**Columns in summary.parquet:**
- `run_id`, `engine`, `workload`, `engine_version`
- `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`
- `qps`, `recall_at_k`, `peak_ram_mb`, `disk_size_mb`
- `status`, `error_message`, `host_fingerprint`

---

## Workloads

| Workload | Description | Metric | Example |
|----------|-------------|--------|---------|
| **A1** | Batch ingest | Throughput (ops/s) | 180 vectors in 60ms = 3000 ops/s |
| **A2** | Pure search (KNN no filters) | Latency, QPS, Recall | 0.016ms/query (FAISS), 1.0 recall |
| **B1** | Search with structured filters | Filtered latency | tag=X: 1.2ms (ChromaDB) |
| **B2** | Continuous ingest | Latency degradation | Queries during inserts |
| **B3** | Persistence/reopen | Restart time | Index reopened in 100ms |
| **B4** | Reopen persisted state | Reopen + search correctness | Disk-bound queries |

---

## Troubleshooting

### Error: "ModuleNotFoundError: chonkie"

```bash
uv sync --dev
```

### Error: "HuggingFace token required"

```bash
huggingface-cli login
# or
export HF_TOKEN="hf_xxxxx"
```

### Error: ".cache/cohere_wikipedia/records.parquet not found"

Let first execution prepare the dataset:

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
# ~10-30 min first time (download), then uses cache
```

### Error: "timeout after 1800s in workload_a1"

Reduce sample_size or increase timeout:

```toml
[dataset]
sample_size = 100_000  # instead of 1M

# or

[timeouts]
ingest_batch_seconds = 3600
```

### Error: "ChromaDB collection already exists"

```bash
rm -rf .benchmarks/projects/benchmark_default/
uv run python main.py run-all --config configs/first-benchmark.toml
```

---

## Additional Documentation

- **Technical specification:** `benchvec/` (modules) and `tests/unit/` (tests)
- **Portuguese guide:** See [COMO_USAR.md](./COMO_USAR.md)
- **Main README:** See [../README.md](../README.md)


---

## Expansion Workloads (C1, C2, D1, Sweep)

### C1 — Concurrent Search
Executes the query set under a thread ladder (`concurrent_workers = [1, 2, 4, 8]`),
reporting QPS and p95 latency per level. Use it to size RAG services for
multiple users:

```bash
uv run python main.py run --project my-benchmark --workload c1
```

### C2 — Dataset Scaling Ladder
Ingests and measures nested dataset prefixes (`sample_sizes = [500, 1000, 2000, 3000]`),
revealing how each engine scales. Prefixes come from one deterministic ordering,
so per-size deltas are comparable.

### D1 — Delete & Upsert Lifecycle
Deletes a deterministic 10% of records, validates post-delete recall, then
upserts live-record overwrites plus a slice of resurrections. Surfaces the
delete-time vs storage-reclaim trade-off (FAISS tombstones vs native deletes).

### Sweep — Recall-QPS Pareto Frontier
Sweeps search parameters (FAISS `nprobe`, LanceDB `nprobes` × `refine_factor`)
and records (params, latency, recall) per query. Powers `chart_08_pareto.png`.
ChromaDB is excluded by default because its HNSW `ef` is fixed at collection
creation; enable `[engines.chroma] ef_search_grid_enabled` to opt in (each
grid point rebuilds the collection). TurboVec sweeps
`[workloads.param_sweep] turbovec_bit_width` the same way (each width
rebuilds the quantized index; the sweep runs at `k_values[0]`).

### Statistics
`report --stats` (default on) writes `stats_summary.parquet` (bootstrap CIs,
CV) and `stats_pairwise.parquet` (pairwise Mann-Whitney U with Holm step-down
correction within each workload family). The markdown analysis includes a
significance table when the pairwise parquet exists.

### Disk hygiene
Engine storage is deleted after each run's metrics are captured. Pass
`--keep-engine-storage` to `run`/`run-all` to retain it for inspection.
