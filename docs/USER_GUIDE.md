# BenchVec User Guide — Complete Reference

> **BenchVec** is a Python benchmarking suite for comparing embedded vector databases (FAISS, ChromaDB, LanceDB) with reproducibility and scientific rigor.

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
cd ~/Desktop/SBBD
uv venv --python 3.12 .venv
uv sync --dev
```

### 2. Validate (30 seconds)

```bash
uv run pytest -q
# Expected: 123 passed
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

## Main Benchmark (for SBBD Paper)

Steps to execute the benchmark for SBBD 2026 submission.

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
uv run pytest -q  # Validate: 123 passed
```

### Step 1: Create Project

```bash
uv run python main.py init --project sbbd-2026
# Output: ✓ Project initialized: /path/to/.benchmarks/projects/sbbd-2026
```

### Step 2: Execute Full Benchmark

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
```

**What will be tested:**
- **Engines**: FAISS (IndexFlatL2), ChromaDB (HNSW), LanceDB (Lance)
- **Dataset**: Cohere Wikipedia 22-12 (180 passages, 768 dimensions)
- **Workloads**: A1-B4 (ingest, search, filtered, persistence, reopen, RAM pressure)
- **Repetitions**: 10 reps × 3 engines = 30 runs

### Step 3: Consolidate Results

```bash
uv run python main.py report \
  --input .benchmarks/projects/sbbd-2026/results \
  --output .benchmarks/projects/sbbd-2026/reports/final.parquet
```

Export to CSV (Excel compatible):

```bash
uv run python main.py report \
  --input .benchmarks/projects/sbbd-2026/results \
  --output .benchmarks/projects/sbbd-2026/reports/final.csv
```

### Step 4: Generate Charts

```bash
uv run python main.py charts --project sbbd-2026
# Output: Generated 6 charts in .benchmarks/projects/sbbd-2026/reports/charts/
```

**Output**: 6 PNGs in `.benchmarks/projects/sbbd-2026/reports/charts/`

### Step 5: Prepare Documentation

Record environment fingerprint for the paper:

```bash
# View host fingerprint (CPU, RAM, storage, OS)
cat .benchmarks/projects/sbbd-2026/results/faiss/a1/run-*/env.json | jq '.host_fingerprint'

# View library versions
cat .benchmarks/projects/sbbd-2026/results/**/env.json | jq '.library_versions'
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
cd /Users/leandro.ferreira/Desktop/SBBD
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
# Expected: 123 passed
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
- **Dimension:** 768 (model `embed-multilingual-v2.0`)
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

### Recommendation for SBBD Paper

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
│   └── charts/               # 6 PNG charts
│       ├── chart_01_latency.png
│       ├── chart_02_qps.png
│       ├── chart_03_recall.png
│       ├── chart_04_peak_ram.png
│       ├── chart_05_disk_footprint.png
│       └── chart_06_build_time.png
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
SBBD/
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
│       ├── chromadb_engine.py   # ChromaDB adapter
│       └── lancedb_engine.py    # LanceDB adapter
├── configs/                     # Configuration templates
│   └── first-benchmark.toml     # Official benchmark config
├── docs/
│   ├── USER_GUIDE.md           # This file (English guide)
│   └── COMO_USAR.md            # Portuguese guide
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest configuration
│   └── unit/                    # 123 unit tests
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
│       │   │   └── charts/      # 6 PNG comparison charts
│       │   │       ├── chart_01_latency.png
│       │   │       ├── chart_02_qps.png
│       │   │       ├── chart_03_recall.png
│       │   │       ├── chart_04_peak_ram.png
│       │   │       ├── chart_05_disk_footprint.png
│       │   │       └── chart_06_build_time.png
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
| **[engines]** | `enabled` | list | [faiss, chromadb, lancedb] | Engines to test |
| | `distance_metric` | enum | l2 | Metric: l2 or cosine |
| | `faiss_index_type` | enum | flat | FAISS: flat or ivfflat |
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

After running a benchmark, automatically generate 6 comparison PNG charts.

### Basic Command

```bash
# Generate all 6 charts
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

### Export for Presentation

```bash
# Copy charts to presentation folder
cp .benchmarks/projects/my-benchmark/reports/charts/*.png ~/presentation/

# In LaTeX/Markdown
![Latency comparison](./charts/chart_01_latency.png)
```

---

## Quick Analysis Generator (90 Records)

For rapid validation of the analysis pipeline and benchmarking workflow:

### Basic Usage

```bash
# Generate detailed markdown report (default)
uv run python test_cohere_90.py
# Output: report.markdown (~1KB, ~25 lines)
```

### Output Formats

Generate analysis in different formats:

```bash
# Markdown format (readable, suitable for documentation)
uv run python test_cohere_90.py --format markdown

# JSON format (structured data for programmatic processing)
uv run python test_cohere_90.py --format json --output analysis.json

# HTML format (self-contained, browser-viewable)
uv run python test_cohere_90.py --format html --output report.html
```

### Report Types

Select the level of detail in the report:

```bash
# Executive summary (high-level overview for presentations)
uv run python test_cohere_90.py --report-type executive_summary

# Detailed (full analysis with comprehensive tables)
uv run python test_cohere_90.py --report-type detailed

# Statistical only (metrics and statistics only)
uv run python test_cohere_90.py --report-type statistical_only
```

### Combined Examples

```bash
# Executive summary in JSON for automated processing
uv run python test_cohere_90.py \
  --report-type executive_summary \
  --format json \
  --output summary.json

# Detailed HTML report for stakeholder presentation
uv run python test_cohere_90.py \
  --report-type detailed \
  --format html \
  --output presentation_report.html

# Statistical analysis in markdown for documentation
uv run python test_cohere_90.py \
  --report-type statistical_only \
  --format markdown \
  --output stats.md
```

### Data Characteristics

The 90-record generator creates realistic synthetic Cohere data:

- **Total records:** 90
- **Engines:** Balanced distribution of FAISS, ChromaDB, LanceDB
- **Workloads:** All workload types (A1-A2, B1-B4) included
- **Metrics statistics:**
  - Mean latency: ~20.7ms
  - QPS (throughput): ~91 queries/second
  - Recall@K: ~0.9165 (91.65%)
  - Peak RAM: ~190-200MB
- **Execution time:** < 1 second
- **Report sizes:**
  - Markdown: ~1KB
  - JSON: ~1.1KB
  - HTML: ~1KB

### Use Cases

1. **Development & Testing** — Validate analysis pipeline during development
2. **CI/CD Pipelines** — Fast automated testing (< 1 second)
3. **Demonstrations** — Show functionality without long computation times
4. **Documentation** — Generate example outputs for user guides
5. **Prototyping** — Quickly test new analysis features

### Integration with Full Benchmarks

The 90-record format is compatible with the full benchmark results:

```bash
# After running main.py run-all, generate quick analysis
uv run python main.py analyze --project benchmark_default --format markdown

# Or keep using the 90-record generator for quick iterations
uv run python test_cohere_90.py --report-type detailed --format html
```

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
| **B4** | Memory pressure | Overflow behavior | Disk-bound queries |

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
