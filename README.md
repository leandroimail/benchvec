# BenchVec: Reproducible Vector Database Benchmarking Suite

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com)
[![Tests](https://img.shields.io/badge/tests-123%20passed-blue)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

BenchVec is a Python benchmarking suite for **fair and reproducible comparison** of embedded vector databases (FAISS, ChromaDB, LanceDB) with rigorous scientific methodology.

**[Português](#português-readme) | [Documentation](#documentation) | [Quick Start](#quick-start)**

---

## Overview

### Problem

Choosing between vector database engines for local AI applications (RAG, assistants, knowledge bases) requires rigorous, reproducible benchmarking:

- **Performance variance** across different hardware, configurations, and workloads
- **Unfair comparisons** between indexing libraries (FAISS) and full-featured databases (ChromaDB, LanceDB)
- **Missing traceability** - no record of individual measurements or anomalies
- **Reproducibility challenges** - seed-based reproducibility is critical for scientific papers

### Solution

BenchVec provides:

✅ **Methodologically sound benchmark design:**
  - Separate workloads for pure ANN and filtered search with metadata
  - Complete hardware fingerprinting (CPU, RAM, storage type)
  - Configurable timeouts, warm-up queries, and repetitions
  - Ground truth validation for Recall@K metrics

✅ **Rich measurement data:**
  - Aggregated metrics (latency, QPS, recall, memory, disk)
  - **Raw micro-data** for every query and ingestion batch
  - Full traceability via Parquet-based storage
  - Environment capture (OS, Python version, library versions)

✅ **Automated reporting:**
  - 6 professional comparison charts (PNG)
  - Parquet/CSV exports for analysis
  - Per-run inspection via CLI
  - Consolidation across multiple runs

✅ **Reproducibility:**
  - Deterministic dataset sampling (with seed control)
  - Pre-computed embeddings (Cohere Wikipedia with fixed dimensions)
  - Exact configuration logging
  - Example configs for common scenarios

---

## Key Features

| Feature | Benefit |
|---------|---------|
| **Complete workload coverage** | A1-B4: ingestion, pure search, filtered search, persistence, memory pressure |
| **Official datasets** | SIFT1M, GIST1M, Cohere Wikipedia 22-12 with pre-computed embeddings |
| **Metadata-aware** | Synthetic but deterministic metadata for realistic RAG scenarios |
| **Timeout policies** | Configurable per-phase timeouts with partial result capture |
| **Multi-engine** | FAISS, ChromaDB, LanceDB with pluggable architecture |
| **Concurrent testing** | Optional multi-threaded query execution |
| **Project organization** | Isolated benchmarks with structured directory layouts |
| **Visual reports** | 6 automatic comparison charts (latency, QPS, recall, RAM, disk, build time) |

---

## Quick Start

### Installation

```bash
git clone https://github.com/leandro-ferreira/SBBD.git
cd SBBD

# Python 3.10+ with uv
uv sync --dev

# Validate
uv run pytest -q
# Expected: 123 passed
```

### First Benchmark

```bash
# Authenticate HuggingFace (for Cohere Wikipedia dataset)
huggingface-cli login

# Run complete benchmark
uv run python main.py run-all --config configs/first-benchmark.toml

# View results
RUNID=$(ls .benchmarks/projects/benchmark_default/results/*/a1/run-* | head -1 | xargs basename)
uv run python main.py inspect --run-id $RUNID --summary
```

### Fast Test Profile (<5 minutes)

For quick validation and development iteration, use the fast benchmark config with max 5,000 entries and batch size 1,000:

```bash
# Fast benchmark (no HuggingFace auth needed - uses public dataset)
uv run python main.py run-all --config configs/fast-benchmark.toml

# Initialize project with fast profile
uv run python main.py init --project fast-test --template configs/fast-benchmark.toml
uv run python main.py run-all --project fast-test
```

**Dataset:** `not-lain/wikipedia-small-3000-embedded` — 3,000 Wikipedia records with pre-computed embeddings (public, no authentication required).

| Parameter | Full Profile | Fast Test |
|-----------|-------------|-----------|
| sample_size | 100,000 | **5,000** |
| ingest_batch_sizes | [1K, 10K, 100K] | **[1K]** |
| repetitions | 10 | **3** |
| warmup_queries | 100 | **10** |
| Target time | Hours | **<5 min** |

### Quick Analysis Generator (90 Records)

For rapid testing and validation of the analysis pipeline:

```bash
# Generate detailed markdown report with 90 synthetic records
uv run python test_cohere_90.py
# Output: report.markdown (~1KB) with complete analysis

# Generate JSON analysis for programmatic processing
uv run python test_cohere_90.py --format json --output analysis.json

# Generate HTML report with interactive statistics
uv run python test_cohere_90.py --format html --output report.html

# All report types (executive summary, detailed, statistical only)
uv run python test_cohere_90.py --report-type detailed --format markdown
uv run python test_cohere_90.py --report-type executive_summary --format json
uv run python test_cohere_90.py --report-type statistical_only --format html
```

**Features:**
- 90 synthetic Cohere records (balanced across engines: FAISS, ChromaDB, LanceDB)
- Execution time: < 1 second (ideal for testing)
- Realistic metrics: latency ~20ms, QPS ~90, recall ~0.91
- Multiple output formats: Markdown, JSON, HTML
- Complete performance analysis in compact form

### Generate Analysis Reports

```bash
# Generate markdown report from full benchmark run
uv run python main.py analyze --project benchmark_default --output analysis.md

# Generate JSON analysis for programmatic processing
uv run python main.py analyze --project benchmark_default --format json --output analysis.json

# Generate HTML report with detailed statistics
uv run python main.py analyze --project benchmark_default --format html --output analysis.html
```

### Generate Comparison Charts

```bash
uv run python main.py charts --project benchmark_default

# Open results
open .benchmarks/projects/benchmark_default/reports/charts/
```

**Expected outputs:**
- `chart_01_latency.png` — Latency (mean/p95/p99) by engine
- `chart_02_qps.png` — Throughput (queries/second)
- `chart_03_recall.png` — Recall@K (target: 1.0)
- `chart_04_peak_ram.png` — Peak memory usage
- `chart_05_disk_footprint.png` — Disk storage per engine
- `chart_06_build_time.png` — Index build + time-to-serve

---

## Documentation

### For Users

- **[USER_GUIDE.md](docs/USER_GUIDE.md)** — Complete English user guide
  - Installation, configuration, all CLI commands
  - Dataset descriptions (SIFT1M, Cohere Wikipedia, Wikipedia local)
  - Workload explanations
  - Troubleshooting
  - Best practices for benchmarking

- **[COMO_USAR.md](docs/COMO_USAR.md)** — Guia completo em português
  - Instalação, configuração, todos os comandos CLI
  - Descrição de datasets
  - Explicação dos workloads
  - Troubleshooting

### For Researchers / SBBD Submission

- **Run full benchmark** → see [Main Benchmark](docs/USER_GUIDE.md#main-benchmark-for-sbbd-paper) (English) or [Benchmark Principal](docs/COMO_USAR.md#benchmark-principal) (português)
- **Official datasets** → see [Official Datasets](docs/USER_GUIDE.md#official-datasets) (English) or [Datasets Oficiais](docs/COMO_USAR.md#datasets-oficiais) (português)
- **Workload definitions** → see [Workloads](docs/USER_GUIDE.md#workloads) section in USER_GUIDE.md
- **Interpretation guide** → see [Generating Charts](docs/USER_GUIDE.md#generating-charts) (English) or [Gerando Gráficos](docs/COMO_USAR.md#gerando-gráficos) (português)

---

## Project Structure

```
SBBD/
├── README.md                     # This file (English)
├── README_pt-BR.md              # Portuguese README
├── benchvec/                    # Main package
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
│   └── engines/                 # Engine adapters
│       ├── base.py
│       ├── faiss_engine.py
│       ├── chromadb_engine.py
│       └── lancedb_engine.py
├── configs/                     # Example configurations
│   ├── first-benchmark.toml     # Official benchmark
│   └── fast-benchmark.toml      # Fast test profile (<5 min)
│   └── fast-benchmark.toml      # Fast test profile (<5 min)
├── docs/
│   ├── COMO_USAR.md            # Guide (Portuguese)
│   └── README_pt-BR.md          # README (Portuguese)
├── tests/
│   ├── unit/                    # 123 unit tests
│   └── conftest.py
└── .benchmarks/                 # Results (auto-created)
```

---

## Requirements

- **Python:** 3.10+
- **OS:** Linux, macOS, Windows (WSL2)
- **RAM:** 4+ GB (8+ GB recommended for full SIFT1M)
- **Disk:** 10+ GB (for cache and results)
- **HuggingFace:** Account + authentication (for Cohere Wikipedia)

---

## Installation

```bash
# Clone repository
git clone https://github.com/leandro-ferreira/SBBD.git
cd SBBD

# Create environment
uv venv --python 3.12 .venv

# Install dependencies
uv sync --dev

# Validate
uv run pytest -q
# Expected: 123 passed ✅
```

### Authenticate HuggingFace

```bash
huggingface-cli login
# or
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

---

## Contributing

Contributions welcome! Areas:

- New engine adapters (Milvus, Qdrant, Weaviate, etc.)
- Additional datasets
- Performance optimizations
- Documentation improvements
- Interactive visualizations

---

## Citation

If you use BenchVec in your research, please cite:

```bibtex
@inproceedings{benchvec2026,
  title={BenchVec: Reproducible Vector Database Benchmarking},
  author={Ferreira, Leandro and Contributors},
  booktitle={Brazilian Symposium on Database Systems (SBBD)},
  year={2026}
}
```

---

## License

MIT License — see [LICENSE](./LICENSE) file

---

## Português README

Este projeto também está documentado em português:

- 📖 **[README em Português](./README_pt-BR.md)**
- 📚 **[Guia de Uso Completo](./docs/COMO_USAR.md)**

---

**Last Updated:** March 29, 2026  
**Status:** ✅ Stable — Ready for SBBD 2026 submission

```bash
uv run python main.py run --config config.toml --workload a2 --workload b1
uv run python main.py run-all --config config.toml
uv run python main.py report --input .benchmarks/results --output reports/final.parquet
```

## Current Coverage

The current implementation includes:

- configuration loading from YAML and TOML
- host fingerprint collection and timeout helpers
- prepared dataset caching for Cohere Wikipedia and Simple Wikipedia fallback
- deterministic synthetic metadata generation and chunking
- exact ground-truth computation and caching
- engine adapters for FAISS, ChromaDB, and LanceDB
- workload runner methods for A1, A2, B1, B2, B3, and B4
- summary persistence, raw micro-data persistence, run inspection, and report consolidation
- **project-based artifact organization with automatic directory structure**
- CLI commands for `init`, `run`, `run-all`, `report`, `inspect`, and `charts` with optional `--project` flag

## CLI Usage

### Project Mode

```bash
# Initialize project
uv run python main.py init --project my-project

# Run benchmarks in project
uv run python main.py run --project my-project --config config.toml
uv run python main.py run-all --project my-project

# Reports and inspection
uv run python main.py report --project my-project
uv run python main.py charts --project my-project
uv run python main.py inspect --project my-project --run-id run-123
```

### Legacy Mode (Without Projects)

```bash
uv run python main.py run --config config.toml
uv run python main.py run-all --config config.toml
uv run python main.py report --input .benchmarks/results --output reports/final.parquet
uv run python main.py inspect --run-id run-123 --summary
uv run python main.py inspect --run-id run-123 --type queries
uv run python main.py inspect --run-id run-123 --type ingest
```

## Notes

- The benchmark artifacts are written under `.benchmarks/projects/{project_name}/` when using project mode, or `.benchmarks/results` in legacy mode.
- The prepared dataset cache is written under `.cache/<dataset-name>`.
- Cohere Wikipedia access still depends on the required Hugging Face permissions.
- Root files `AGENTS.md`, `PRD03.md`, and `Resumo.md` are kept outside implementation commits unless explicitly requested.

