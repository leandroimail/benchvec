# Como Usar o BenchVec — Guia Completo

> **BenchVec** é a suite de benchmark em Python para comparar bancos de dados vetoriais embarcados (FAISS, ChromaDB, LanceDB) com reproducibilidade e rigor científico.

---

## Índice

1. [Quick Start](#quick-start)
2. [Instalação Detalhada](#instalação)
3. [Datasets Oficiais](#datasets-oficiais)
4. [Sistema de Projetos](#sistema-de-projetos)
5. [Configuração de Benchmark](#configuração)
6. [Executando Benchmarks](#executando)
7. [Gerador Rápido de Análises (90 Registros)](#gerador-rápido-de-análises-90-registros)
8. [Gerando Gráficos](#gerando-gráficos)
9. [Inspecionando Resultados](#inspecionando)
10. [Referência de Comandos](#referência)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Instalar

```bash
cd ~/Desktop/SBBD
uv venv --python 3.12 .venv
uv sync --dev
```

### 2. Validar (30 segundos)

```bash
uv run pytest -q
# Esperado: 123 passed
```

### 3. Primeiro Benchmark 

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
# Captures run_id na saída (ex: run-3e831116cfa4)
```

### 3b. Benchmark Rápido (<5 minutos)

Para validação rápida sem autenticação HuggingFace:

```bash
uv run python main.py run-all --config configs/fast-benchmark.toml
```

**Dataset:** `not-lain/wikipedia-small-3000-embedded` — 3.000 registros com pre-computed embeddings (público, sem HF_TOKEN).

**Templates disponíveis:**
- `configs/fast-benchmark.toml` — Perfil rápido (5K entradas, batch=1000, sem HF_TOKEN)
- `configs/first-benchmark.toml` — Perfil completo (requer HF_TOKEN para Cohere Wikipedia)

### 4. Ver Resultados (instantâneo)

```bash
RUNID="run-3e831116cfa4"  # seu run_id
uv run python main.py inspect --run-id $RUNID --summary
```

**Output esperado:** Tabela com A1-B4 métricas (latência, recall, QPS) por engine.

---

## Benchmark Principal (para Artigo SBBD)

Passos para executar o benchmark que será submetido ao SBBD 2026.

### Pré-requisitos

1. **Autenticar HuggingFace** (Cohere Wikipedia)
```bash
huggingface-cli login
# ou
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

2. **Preparar ambiente** (se não foi feito)
```bash
uv sync --dev
uv run pytest -q  # Validar: 123 passed
```

### Passo 1: Criar Projeto

```bash
uv run python main.py init --project sbbd-2026
# Output: ✓ Project initialized: /path/to/.benchmarks/projects/sbbd-2026
```

### Passo 2: Executar Benchmark Completo

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
```

**O que será testado:**
- **Engines**: FAISS (IndexFlatL2), ChromaDB (HNSW), LanceDB (Lance)
- **Dataset**: Cohere Wikipedia 22-12 (180 passages, 768 dimensões)
- **Workloads**: A1-B4 (ingestão, busca, filtrado, persistência, reabertura, restrição RAM)
- **Repetições**: 10 reps × 3 engines = 30 runs

### Passo 3: Consolidar Resultados

```bash
uv run python main.py report \
  --input .benchmarks/projects/sbbd-2026/results \
  --output .benchmarks/projects/sbbd-2026/reports/final.parquet
```

Exportar também para CSV (compatível com Excel):

```bash
uv run python main.py report \
  --input .benchmarks/projects/sbbd-2026/results \
  --output .benchmarks/projects/sbbd-2026/reports/final.csv
```

### Passo 4: Gerar Gráficos

```bash
uv run python main.py charts --project sbbd-2026
# Output: Generated 6 charts in .benchmarks/projects/sbbd-2026/reports/charts/
```

**Output:** 6 PNG em `.benchmarks/projects/sbbd-2026/reports/charts/`

### Passo 5: Preparar Documentação

Registrar informações de ambiente para o artigo:

```bash
# Ver host fingerprint (CPU, RAM, storage, SO)
cat .benchmarks/projects/sbbd-2026/results/faiss/a1/run-*/env.json | jq '.host_fingerprint'

# Ver versões de bibliotecas
cat .benchmarks/projects/sbbd-2026/results/**/env.json | jq '.library_versions'
```

---

## Instalação

### Requisitos

- **Python:** 3.10+
- **OS:** Linux, macOS, Windows (WSL2)
- **RAM:** 4+ GB
- **Disk:** 10+ GB (para datasets preparados)

### Passo 1: Preparar Ambiente

```bash
# Setup Python 3.12
uv python install 3.12

# Projeto
cd /Users/leandro.ferreira/Desktop/SBBD
uv venv --python 3.12 .venv

# Dependências (dev + production)
uv sync --dev
```

### Passo 2: Validar

```bash
# Linting
uv run ruff check .
# Esperado: 0 errors

# Type checking
uv run mypy benchvec tests
# Esperado: Success

# Testes
uv run pytest -q
# Esperado: 123 passed
```

### Passo 3: Datasets (Opcional)

Para **Cohere Wikipedia** (recomendado):

```bash
# Autenticar (requer account HF)
huggingface-cli login
# Cole token quando solicitado
```

Para **offline**, preparar cache em `.cache/cohere_wikipedia/`:

```
.cache/
└─ cohere_wikipedia/
   ├─ records.parquet
   ├─ selected_ids.json
   ├─ doc_chunk_map.json
   └─ manifest.json
```

---

## Datasets Oficiais

BenchVec suporta múltiplos datasets organizados em dois grupos científicos:

### Grupo A: Benchmark ANN Clássico

Foco em eficiência de busca vetorial pura, sem filtros estruturados.

#### SIFT1M (Recomendado para Artigos)
- **Dimensão:** 128
- **Tamanho:** 1 milhão de vetores
- **Queries:** 10.000
- **Ground truth:** Disponível
- **Uso:** Benchmark canônico em busca vetorial, permite comparação com trabalhos relacionados
- **URL:** http://sift.fi.msu.ru/
- **Performance esperada:** FAISS ~0.016ms (Flat), ChromaDB ~1-2ms, LanceDB ~0.5ms

#### GIST1M
- **Dimensão:** 960
- **Tamanho:** 1 milhão de vetores
- **Queries:** 1.000 (atenção: muitas publicações usam apenas 1k!)
- **Ground truth:** Disponível
- **Nota:** Dimensão alta; p95/p99 menos significativas (n=1000)

### Grupo B: Workload RAG Local com Metadados

Foco em cenários realistas com persistência, filtros estruturados e metadados.

#### Cohere Wikipedia 22-12 (Principal - Recomendado)
- **Corpus:** Simple English Wikipedia com embeddings pré-computados
- **Dimensão:** 768 (modelo `embed-multilingual-v2.0`)
- **Tamanho:** Até 170.000 passages (use sample_size configurável)
- **Pré-requisito:** Autenticação HuggingFace (CLI login)
- **Vantagem:** Embeddings pré-computados = reproduzibilidade garantida
- **Configuração:**
  ```toml
  [dataset]
  primary = "cohere_wikipedia"
  sample_size = 100000  # ou 10000 para teste rápido
  ```

#### Wikipedia 20220301 (Alternativa Pública)
- **Corpus:** Simple English Wikipedia raw text (sem embeddings pré-computados)
- **Dimensão:** 384 (modelo `sentence-transformers/all-MiniLM-L6-v2`)
- **Tamanho:** Ajustável; ~200k chunks após segmentação
- **Vantagem:** Não requer autenticação HuggingFace
- **Pipeline:** Texto → Chunking (TokenChunker, 256 tokens) → Embedding local
- **Configuração:**
  ```toml
  [dataset]
  primary = "simple_wikipedia"
  chunk_size = 256
  chunk_overlap = 32
  ```

#### Metadados Estruturados (Sintéticos)
Todos os datasets do Grupo B incluem metadados para testes de filtrado:
- **categoria**: derivada por palavras-chave (data_engineering, rag, general, etc)
- **tag**: tags de filtro controladas por categoria
- **autor**: distribuição determinística (8 valores)
- **data_criacao**: timestamps sintéticos determinísticos (2021-2025)
- **document_length_bucket**: faixas de tamanho (short/medium/long/xlong)

### Recomendação para Artigo SBBD

1. **Para Benchmark A (ANN Puro):** Usar **SIFT1M** (dataset canônico, não será questionado)
2. **Para Benchmark B (RAG Local):** Usar **Cohere Wikipedia**
3. **Configuração Padrão:**
   ```toml
   [dataset]
   primary = "cohere_wikipedia"
   sample_size = 100000
   ground_truth_k = 100
   ```

---

## Sistema de Projetos

BenchVec organiza cada benchmark em um **projeto** independente com estrutura padronizada.

### O que é um Projeto?

Um projeto é um diretório que agrupa:
- **Configuração** de benchmark (project.toml)
- **Resultados** de todas as execuções
- **Relatórios** consolidados
- **Gráficos** PNG de comparação
- **Logs** de execução

### Estrutura de Projeto

```
.benchmarks/projects/my-project/
├── project.toml              # Metadados do projeto
├── config.toml               # Config default (opcional)
├── results/                  # Resultados de execuções
│   ├── faiss/
│   │   ├── a1/run-xxx/
│   │   ├── a2/run-yyy/
│   │   └── ...
│   ├── chromadb/
│   ├── lancedb/
│   └── index.parquet
├── reports/                  # Relatórios consolidados
│   ├── final.parquet
│   └── charts/               # 6 gráficos PNG
│       ├── chart_01_latency.png
│       ├── chart_02_qps.png
│       ├── chart_03_recall.png
│       ├── chart_04_peak_ram.png
│       ├── chart_05_disk_footprint.png
│       └── chart_06_build_time.png
└── logs/                     # Logs de execution
```

### Criar um Novo Projeto

```bash
# Criar estrutura padrão
uv run python main.py init --project my-benchmark

# Output esperado:
# ✓ Project initialized: /path/to/.benchmarks/projects/my-benchmark
# Results: /path/to/.benchmarks/projects/my-benchmark/results
# Reports: /path/to/.benchmarks/projects/my-benchmark/reports
# Logs: /path/to/.benchmarks/projects/my-benchmark/logs
```

### Usar um Projeto

```bash
# Rodar benchmark em um projeto específico
uv run python main.py run-all \
  --config configs/first-benchmark.toml \
  --project my-benchmark

# Gerar gráficos do projeto
uv run python main.py charts --project my-benchmark

# Ver resultados
open .benchmarks/projects/my-benchmark/reports/charts/
```

### Projetos Padrão

- **benchmark_default**: Projeto padrão quando nenhum é especificado
- Criado automaticamente na primeira execução

---

## Estrutura do Projeto

A raiz do projeto e os projetos de benchmark seguem esta hierarquia:

```
SBBD/
├── README.md                     # Documentação principal (Inglês)
├── README_pt-BR.md              # Documentação principal (Português)
├── LICENSE                       # Licença MIT
├── benchvec/                    # Pacote principal
│   ├── __init__.py
│   ├── cli.py                   # Comandos CLI
│   ├── config.py                # Configuração
│   ├── datasets.py              # Carregamento de datasets
│   ├── chunker.py               # Segmentação de texto
│   ├── ground_truth.py          # Ground truth
│   ├── metrics.py               # Medições
│   ├── runner.py                # Orquestração de benchmark
│   ├── raw_store.py             # Armazenamento de micro-dados
│   ├── report.py                # Relatórios e gráficos
│   ├── host.py                  # Fingerprinting de hardware
│   ├── project.py               # Gerenciamento de projetos
│   ├── timeouts.py              # Políticas de timeout
│   └── engines/                 # Adaptadores de engines
│       ├── __init__.py
│       ├── base.py              # Classe base de engine
│       ├── faiss_engine.py      # Adaptador FAISS
│       ├── chromadb_engine.py   # Adaptador ChromaDB
│       └── lancedb_engine.py    # Adaptador LanceDB
├── configs/                     # Templates de configuração
│   └── first-benchmark.toml     # Config de benchmark oficial
├── docs/
│   ├── USER_GUIDE.md           # Guia em inglês
│   └── COMO_USAR.md            # Este arquivo (guia em português)
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Configuração pytest
│   └── unit/                    # 123 testes unitários
│       ├── test_cli.py
│       ├── test_config.py
│       ├── test_datasets.py
│       ├── test_*.py
│       └── ...
├── .benchmarks/                # Diretório de resultados (auto-criado)
│   └── projects/
│       ├── benchmark_default/
│       │   ├── project.toml
│       │   ├── results/         # Resultados de execuções
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
│       │   ├── reports/         # Relatórios consolidados
│       │   │   ├── final.parquet
│       │   │   ├── final.csv
│       │   │   └── charts/      # 6 gráficos PNG de comparação
│       │   │       ├── chart_01_latency.png
│       │   │       ├── chart_02_qps.png
│       │   │       ├── chart_03_recall.png
│       │   │       ├── chart_04_peak_ram.png
│       │   │       ├── chart_05_disk_footprint.png
│       │   │       └── chart_06_build_time.png
│       │   └── logs/            # Logs de execução
│       └── meu-projeto/         # Projeto customizado
│           └── [mesma estrutura]
├── .cache/                      # Cache de datasets
│   ├── cohere_wikipedia/
│   ├── simple_wikipedia/
│   └── ...
└── .venv/                       # Ambiente virtual Python
```

---

## Configuração

### Arquivo TOML

Benchmarks configurados via TOML (ex: [configs/first-benchmark.toml](../configs/first-benchmark.toml)).

#### Exemplo Mínimo

```toml
project_name = "meu_benchmark"
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

#### Campos-Chave

| Seção | Campo | Tipo | Padrão | Descrição |
|-------|-------|------|--------|-----------|
| — | `project_name` | string | benchvec | Nome do projeto |
| — | `seed` | int | 42 | Semente reproducibilidade |
| **[paths]** | `benchmark_dir` | path | .benchmarks | Resultados |
| | `cache_dir` | path | .cache | Cache de datasets |
| **[dataset]** | `primary` | enum | cohere_wikipedia | Dataset: cohere_wikipedia, simple_wikipedia, sift1m, gist1m |
| | `sample_size` | int | 100_000 | Número de vetores |
| | `ground_truth_k` | int | 100 | K para ground-truth |
| **[engines]** | `enabled` | list | [faiss, chromadb, lancedb] | Engines a testar |
| | `distance_metric` | enum | l2 | Métrica: l2 ou cosine |
| | `faiss_index_type` | enum | flat | FAISS: flat ou ivfflat |
| **[workloads]** | `repetitions` | int | 10 | Reps por engine |
| | `k_values` | list | [10] | Valores K para recall@K |
| | `ingest_batch_sizes` | list | [1000] | Tamanhos de batch |

### Configs Pré-prontas

**Quick Test:**
```toml
[dataset]
sample_size = 50
[engines]
enabled = ["faiss", "lancedb"]
[workloads]
repetitions = 2
```

**Full Suite (1-2 horas):** Use `configs/first-benchmark.toml`

---

## Executando

### run-all

Executa todos os workloads (A1-B4) com todas as engines:

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
```

**Output:**
```
.benchmarks/projects/benchmark_default/results/
├─ faiss/a1/run-238f968acf79/
│  ├─ summary.parquet       # Métricas
│  ├─ env.json              # Host info
│  └─ raw_ingest_*.parquet  # Micro-dados
├─ faiss/a2/run-.../
├─ faiss/b1-b4/.../
├─ chromadb/...
└─ lancedb/...
```


### run

Workloads selecionados:

```bash
# Apenas A1 e A2
uv run python main.py run \
  --config configs/first-benchmark.toml \
  --workload a1 --workload a2

# Apenas FAISS
uv run python main.py run \
  --config configs/first-benchmark.toml \
  --engine faiss
```

---

## Gerador Rápido de Análises (90 Registros)

Para validação rápida do pipeline de análise e workflow de benchmarking:

### Uso Básico

```bash
# Gerar relatório Markdown detalhado (padrão)
uv run python test_cohere_90.py
# Saída: report.markdown (~1KB, ~25 linhas)
```

### Formatos de Saída

Gere análise em diferentes formatos:

```bash
# Formato Markdown (legível, adequado para documentação)
uv run python test_cohere_90.py --format markdown

# Formato JSON (dados estruturados para processamento programático)
uv run python test_cohere_90.py --format json --output analysis.json

# Formato HTML (auto-contido, visível em navegador)
uv run python test_cohere_90.py --format html --output report.html
```

### Tipos de Relatório

Selecione o nível de detalhe no relatório:

```bash
# Resumo executivo (visão de alto nível para apresentações)
uv run python test_cohere_90.py --report-type executive_summary

# Detalhado (análise completa com tabelas abrangentes)
uv run python test_cohere_90.py --report-type detailed

# Apenas estatístico (métricas e estatísticas apenas)
uv run python test_cohere_90.py --report-type statistical_only
```

### Exemplos Combinados

```bash
# Resumo executivo em JSON para processamento automatizado
uv run python test_cohere_90.py \
  --report-type executive_summary \
  --format json \
  --output summary.json

# Relatório HTML detalhado para apresentação a stakeholders
uv run python test_cohere_90.py \
  --report-type detailed \
  --format html \
  --output presentation_report.html

# Análise estatística em Markdown para documentação
uv run python test_cohere_90.py \
  --report-type statistical_only \
  --format markdown \
  --output stats.md
```

### Características dos Dados

O gerador de 90 registros cria dados sintéticos realísticos do Cohere:

- **Total de registros:** 90
- **Engines:** Distribuição balanceada de FAISS, ChromaDB, LanceDB
- **Workloads:** Todos os tipos de workload incluídos (A1-A2, B1-B4)
- **Estatísticas de métricas:**
  - Latência média: ~20.7ms
  - QPS (throughput): ~91 queries/segundo
  - Recall@K: ~0.9165 (91.65%)
  - Pico de RAM: ~190-200MB
- **Tempo de execução:** < 1 segundo
- **Tamanhos de relatório:**
  - Markdown: ~1KB
  - JSON: ~1.1KB
  - HTML: ~1KB

### Casos de Uso

1. **Desenvolvimento e Testes** — Valide o pipeline de análise durante o desenvolvimento
2. **Pipelines CI/CD** — Testes automatizados rápidos (< 1 segundo)
3. **Demonstrações** — Mostre funcionalidade sem tempos de computação longos
4. **Documentação** — Gere exemplos de output para guias de usuário
5. **Prototipagem** — Teste rapidamente novas features de análise

### Integração com Benchmarks Completos

O formato de 90 registros é compatível com os resultados de benchmark completos:

```bash
# Depois de rodar main.py run-all, gere análise rápida
uv run python main.py analyze --project benchmark_default --format markdown

# Ou continue usando o gerador de 90 registros para iterações rápidas
uv run python test_cohere_90.py --report-type detailed --format html
```

---

## Gerando Gráficos

Após executar um benchmark, gerar 6 gráficos PNG de comparação automaticamente.

### Comando Básico

```bash
# Gerar todos os 6 gráficos
uv run python main.py charts --project my-benchmark
```

### Output

Os gráficos são salvos em `.benchmarks/projects/my-benchmark/reports/charts/`:

```
chart_01_latency.png            # Latência mean/p95/p99 por engine
chart_02_qps.png                # Throughput (queries por segundo)
chart_03_recall.png             # Recall@K por engine (target: 1.0)
chart_04_peak_ram.png           # Pico de RAM (MB) por workload
chart_05_disk_footprint.png     # Footprint disco agregado
chart_06_build_time.png         # Build time + time-to-serve
```

**Tamanho total:** ~240-250 KB (40-50 KB cada)

### Customizar Output

```bash
# Especificar diretório de entrada
uv run python main.py charts \
  --project my-benchmark \
  --input .benchmarks/projects/my-benchmark/results

# Especificar diretório de saída
uv run python main.py charts \
  --project my-benchmark \
  --output .benchmarks/projects/my-benchmark/reports/v2_charts/
```

### Interpretar os Gráficos

#### Chart 1: Latência
- **Eixo Y:** Latência em milissegundos (log recomendada)
- **Barras:** Mean com intervalos de confiança (95%)
- **Cores:** FAISS (azul), ChromaDB (laranja), LanceDB (verde)
- **Esperado:** FAISS ~0.01-0.05ms, ChromaDB/LanceDB ~1-10ms

#### Chart 2: QPS (Queries Per Second)
- **Eixo Y:** Throughput em queries/segundo
- **Escalas:** Podem diferir 10-100x entre engines
- **Esperado:** FAISS ~50k+, ChromaDB ~500-1k, LanceDB ~600-750

#### Chart 3: Recall@K
- **Eixo Y:** Recall (0.0 a 1.0)
- **Esperado:** 1.0 em todos engines (todos encontram top-K corretos)
- **Alerta:** Se < 1.0, verificar ground truth ou bugs

#### Chart 4: Peak RAM
- **Eixo Y:** RAM em MB
- **Workloads:** A1 (ingestão) vs A2-B4 (queries)
- **Esperado:** 300-500 MB para dataset 100k

#### Chart 5: Disk Footprint
- **Eixo Y:** Tamanho em MB (agregado)
- **Comparação:** ChromaDB > FAISS > LanceDB
- **Esperado:** LanceDB ~0.05-0.2 MB, ChromaDB ~1-3 MB

#### Chart 6: Build Time + Time-to-Serve
- **Eixo Y:** Tempo em segundos
- **Fases:** Build time (azul) vs Time-to-serve (laranja)
- **Esperado:** FAISS ~0.0-0.01s, ChromaDB ~0.02-0.05s

### Exportar para Apresentação

```bash
# Copiar charts para pasta de apresentação
cp .benchmarks/projects/my-benchmark/reports/charts/*.png ~/presentation/

# Em LaTeX/Markdown
![Latency comparison](./charts/chart_01_latency.png)
```

---

### inspect

Detalhes de um run:

```bash
# Resumo agregado
uv run python main.py inspect --run-id run-3e831116cfa4 --summary

# Queries individuais
uv run python main.py inspect --run-id run-3e831116cfa4 --type queries

# Ingestão
uv run python main.py inspect --run-id run-3e831116cfa4 --type ingest
```

### report

Consolidar em uma tabela:

```bash
# Parquet (análise em Python)
uv run python main.py report \
  --input .benchmarks/projects/benchmark_default/results \
  --output reports/consolidated.parquet

# CSV (Excel)
uv run python main.py report \
  --input .benchmarks/projects/benchmark_default/results \
  --output reports/consolidated.csv
```

### Ler em Python

```python
import pandas as pd

# Tabela consolidada
df = pd.read_parquet('reports/consolidated.parquet')

# Filtrar
faiss_a2 = df[(df['engine'] == 'faiss') & (df['workload'] == 'a2')]
print(faiss_a2[['mean_latency_ms', 'qps', 'recall_at_k']])

# Micro-dados brutos
queries = pd.read_parquet('.benchmarks/.../raw_queries_run-XXX.parquet')
```

---

## Referência Completa de Comandos

```bash
# Run-all (suite padrão)
uv run python main.py run-all --config configs/first-benchmark.toml

# Run selecionado
uv run python main.py run --config CONFIG.toml --workload a1 --engine faiss

# Report consolidado
uv run python main.py report --input .benchmarks/.../results --output reports/final.parquet

# Inspecionar
uv run python main.py inspect --run-id run-XXX --type summary|queries|ingest

# Init (novo projeto)
uv run python main.py init --name my-project
```

---

## Estrutura de Output

```
.benchmarks/projects/benchmark_default/results/
├─ faiss/
│  ├─ a1/run-238f968acf79/
│  │  ├─ summary.parquet           # Métricas: latência, QPS, recall
│  │  ├─ env.json                  # CPU, RAM, storage, Python version
│  │  └─ raw_ingest_*.parquet      # Cada ingestão individual
│  ├─ a2/run-f2a82d84b734/
│  │  ├─ summary.parquet
│  │  ├─ env.json
│  │  └─ raw_queries_*.parquet     # Cada query individual
│  ├─ b1-b4/...
├─ chromadb/...
├─ lancedb/...
└─ index.parquet                    # Índice: run_id → summary_path
```

**Colunas de summary.parquet:**
- `run_id`, `engine`, `workload`, `engine_version`
- `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`
- `qps`, `recall_at_k`, `peak_ram_mb`, `disk_size_mb`
- `status`, `error_message`, `host_fingerprint`

---

## Workloads

| Workload | Descrição | Métrica | Exemplo |
|----------|-----------|---------|---------|
| **A1** | Ingestão em lote | Throughput (ops/s) | 180 vetores em 60ms = 3000 ops/s |
| **A2** | Busca pura (KNN sem filtros) | Latência, QPS, Recall | 0.016ms/query (FAISS), 1.0 recall |
| **B1** | Busca com filtros estruturados | Latência com filtro | tag=X: 1.2ms (ChromaDB) |
| **B2** | Ingestão contínua | Degradação latência | Queries durante inserts |
| **B3** | Persistência/reabertura | Tempo para restart | Índice reaberto em 100ms |
| **B4** | Restrição de memória | Comportamento overflow | Disk-bound queries |

---

## Troubleshooting

### Erro: "ModuleNotFoundError: chonkie"

```bash
uv sync --dev
```

### Erro: "HuggingFace token required"

```bash
huggingface-cli login
# ou
export HF_TOKEN="hf_xxxxx"
```

### Erro: ".cache/cohere_wikipedia/records.parquet not found"

Deixe a primeira execução preparar o dataset:

```bash
uv run python main.py run-all --config configs/first-benchmark.toml
# ~10-30 min primeira vez (download), depois usa cache
```

### Erro: "timeout after 1800s in workload_a1"

Reduzir sample_size ou aumentar timeout:

```toml
[dataset]
sample_size = 100_000  # ao invés de 1M

# ou

[timeouts]
ingest_batch_seconds = 3600
```

### Erro: "ChromaDB collection already exists"

```bash
rm -rf .benchmarks/projects/benchmark_default/
uv run python main.py run-all --config configs/first-benchmark.toml
```

---

## Documentação Adicional

- **Especificação técnica:** `benchvec/` (modules) e `tests/unit/` (testes)
- **Exemplos de configs:** `configs/` (first-benchmark.toml, etc)
