# BenchVec: Suite de Benchmarking Reproduzível para Bancos de Dados Vetoriais

[![Status](https://img.shields.io/badge/status-estável-brightgreen)](https://github.com)
[![Testes](https://img.shields.io/badge/testes-123%20aprovados-blue)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab)](https://python.org)
[![Licença](https://img.shields.io/badge/licence-MIT-green)](https://opensource.org/licenses/MIT)

BenchVec é uma suite em Python para **comparação justa e reproduzível** de bancos de dados vetoriais embarcados (FAISS, ChromaDB, LanceDB) com rigorous metodologia científica.

**[English](./README.md) | [Documentação](#documentação) | [Quick Start](#quick-start) | [SBBD 2026](#sbbd-2026)**

---

## Visão Geral

### Problema

Escolher entre engines de bancos de dados vetoriais para aplicações locais baseadas em IA (RAG, assistentes, bases de conhecimento) requer benchmarking rigoroso e reproduzível:

- **Variância de desempenho** entre diferentes hardware, configurações e workloads
- **Comparações injustas** entre bibliotecas de indexação (FAISS) e bancos completos (ChromaDB, LanceDB)
- **Falta de rastreabilidade** — sem registro de medições individuais ou anomalias
- **Desafios de reproducibilidade** — reproducibilidade baseada em seed é crítica para papers científicos

### Solução

BenchVec fornece:

✅ **Design metodologicamente correto:**
  - Workloads separados para ANN puro e busca filtrada com metadados
  - Fingerprinting completo de hardware (CPU, RAM, tipo de storage)
  - Timeouts configuráveis, queries de aquecimento, repetições
  - Validação de ground truth para métricas Recall@K

✅ **Dados ricos de medição:**
  - Métricas agregadas (latência, QPS, recall, memória, disco)
  - **Micro-dados brutos** para cada query e lote de ingestão
  - Rastreabilidade completa via armazenamento Parquet
  - Captura de ambiente (SO, versão Python, versões de biblioteca)

✅ **Relatórios automatizados:**
  - 6 gráficos profissionais de comparação (PNG)
  - Exports em Parquet/CSV para análise
  - Inspeção por-run via CLI
  - Consolidação entre múltiplas execuções

✅ **Reproducibilidade:**
  - Amostragem determinística de dataset (com controle de seed)
  - Embeddings pré-computados (Cohere Wikipedia com dimensões fixas)
  - Logging exato de configuração
  - Configs exemplo para cenários comuns

---

## Características Principais

| Característica | Benefício |
|---|---|
| **Cobertura completa de workloads** | A1-B4: ingestão, busca pura, busca filtrada, persistência, pressão de memória |
| **Datasets oficiais** | SIFT1M, GIST1M, Cohere Wikipedia 22-12 com embeddings pré-computados |
| **Metadados-consciente** | Metadados sintéticos mas determinísticos para cenários RAG realistas |
| **Políticas de timeout** | Configurável por-fase com captura de resultados parciais |
| **Multi-engine** | FAISS, ChromaDB, LanceDB com arquitetura plugável |
| **Testes concorrentes** | Execução de queries opcional multi-thread |
| **Organização de projetos** | Benchmarks isolados com layouts de diretório estruturados |
| **Relatórios visuais** | 6 gráficos automáticos (latência, QPS, recall, RAM, disco, build time) |

---

## Quick Start

### Instalação

```bash
git clone https://github.com/leandro-ferreira/SBBD.git
cd SBBD

# Python 3.10+ com uv
uv sync --dev

# Validar
uv run pytest -q
# Esperado: 123 passed
```

### Primeiro Benchmark

```bash
# Autenticar HuggingFace (para dataset Cohere Wikipedia)
huggingface-cli login

# Rodar benchmark completo
uv run python main.py run-all --config configs/first-benchmark.toml

# Ver resultados
RUNID=$(ls .benchmarks/projects/benchmark_default/results/*/a1/run-* | head -1 | xargs basename)
uv run python main.py inspect --run-id $RUNID --summary
```

### Perfil de Teste Rapido (<5 minutos)

Para validacao rapida e iteracao de desenvolvimento, use a configuracao fast com maximo de 5.000 entradas e batch size 1.000:

```bash
# Benchmark rapido (sem autenticacao HuggingFace - dataset publico)
uv run python main.py run-all --config configs/fast-benchmark.toml

# Inicializar projeto com perfil rapido
uv run python main.py init --project fast-test --template configs/fast-benchmark.toml
uv run python main.py run-all --project fast-test
```

**Dataset:** `not-lain/wikipedia-small-3000-embedded` — 3.000 registros da Wikipedia com embeddings pre-computados (publico, sem autenticacao).

| Parametro | Perfil Completo | Teste Rapido |
|-----------|----------------|--------------|
| sample_size | 100.000 | **5.000** |
| ingest_batch_sizes | [1K, 10K, 100K] | **[1K]** |
| repetitions | 10 | **3** |
| warmup_queries | 100 | **10** |
| Tempo alvo | Horas | **<5 min** |

### Gerador Rapido de Analises (90 Registros)

Para testes rápidos e validação do pipeline de análise:

```bash
# Gerar relatório Markdown detalhado com 90 registros sintéticos
uv run python test_cohere_90.py
# Saída: report.markdown (~1KB) com análise completa

# Gerar análise JSON para processamento programático
uv run python test_cohere_90.py --format json --output analysis.json

# Gerar relatório HTML com estatísticas interativas
uv run python test_cohere_90.py --format html --output report.html

# Todos os tipos de relatório (resumo executivo, detalhado, estatístico)
uv run python test_cohere_90.py --report-type detailed --format markdown
uv run python test_cohere_90.py --report-type executive_summary --format json
uv run python test_cohere_90.py --report-type statistical_only --format html
```

**Características:**
- 90 registros sintéticos Cohere (balanceados entre engines: FAISS, ChromaDB, LanceDB)
- Tempo de execução: < 1 segundo (ideal para testes)
- Métricas realísticas: latência ~20ms, QPS ~90, recall ~0.91
- Múltiplos formatos de saída: Markdown, JSON, HTML
- Análise de performance completa em forma compacta

### Gerar Gráficos de Comparação

```bash
uv run python main.py charts --project benchmark_default

# Abrir resultados
open .benchmarks/projects/benchmark_default/reports/charts/
```

**Outputs esperados:**
- `chart_01_latency.png` — Latência (mean/p95/p99) por engine
- `chart_02_qps.png` — Throughput (queries/segundo)
- `chart_03_recall.png` — Recall@K (alvo: 1.0)
- `chart_04_peak_ram.png` — Pico de memória
- `chart_05_disk_footprint.png` — Armazenamento em disco por engine
- `chart_06_build_time.png` — Build time + time-to-serve

---

## Documentação

### Para Usuários

- **[COMO_USAR.md](docs/COMO_USAR.md)** — Guia completo em português
  - Instalação, configuração, todos os comandos CLI
  - Descrição de datasets (SIFT1M, Cohere Wikipedia, Wikipedia local)
  - Explicação dos workloads
  - Troubleshooting
  - Melhores práticas para benchmarking

- **[USER_GUIDE.md](docs/USER_GUIDE.md)** — Complete English user guide
  - Installation, configuration, all CLI commands
  - Dataset descriptions
  - Workload explanations
  - Troubleshooting

### Para Pesquisadores / Submissão SBBD

- **Rodar benchmark completo** → veja [Benchmark Principal](docs/COMO_USAR.md#benchmark-principal) (português) ou [Main Benchmark](docs/USER_GUIDE.md#main-benchmark-for-sbbd-paper) (English)
- **Datasets oficiais** → veja [Datasets Oficiais](docs/COMO_USAR.md#datasets-oficiais) (português) ou [Official Datasets](docs/USER_GUIDE.md#official-datasets) (English)
- **Definição dos workloads** → veja seção [Workloads](docs/USER_GUIDE.md#workloads) em USER_GUIDE.md ou COMO_USAR.md
- **Guia de interpretação** → veja [Gerando Gráficos](docs/COMO_USAR.md#gerando-gráficos) (português) ou [Generating Charts](docs/USER_GUIDE.md#generating-charts) (English)

---

## Estrutura do Projeto

```
SBBD/
├── README.md                     # Este arquivo (Inglês)
├── README_pt-BR.md              # README em português
├── benchvec/                    # Pacote principal
│   ├── cli.py                   # Comandos CLI
│   ├── config.py                # Configuração
│   ├── datasets.py              # Carregamento de datasets
│   ├── chunker.py               # Segmentação de texto
│   ├── ground_truth.py          # Ground truth
│   ├── metrics.py               # Medições
│   ├── runner.py                # Orquestração de benchmark
│   ├── raw_store.py             # armazenamento de micro-dados
│   ├── report.py                # Relatórios e gráficos
│   ├── host.py                  # Fingerprinting de hardware
│   ├── project.py               # Gerenciamento de projetos
│   └── engines/                 # Adaptadores de engines
│       ├── base.py
│       ├── faiss_engine.py
│       ├── chromadb_engine.py
│       └── lancedb_engine.py
├── configs/                     # Configurações exemplo
│   └── first-benchmark.toml     # Benchmark oficial
│   └── fast-benchmark.toml      # Perfil de teste rapido (<5 min)
├── docs/
│   ├── COMO_USAR.md            # Guia em português
│   └── README_pt-BR.md          # Este arquivo
├── tests/
│   ├── unit/                    # 123 testes unitários
│   └── conftest.py
└── .benchmarks/                 # Resultados (auto-criado)
```

---

## Requisitos

- **Python:** 3.10+
- **SO:** Linux, macOS, Windows (WSL2)
- **RAM:** 4+ GB (8+ GB recomendado para SIFT1M completo)
- **Disco:** 10+ GB (para cache e resultados)
- **HuggingFace:** Conta + autenticação (para Cohere Wikipedia)

---

## Instalação

```bash
# Clone do repositório
git clone https://github.com/leandro-ferreira/SBBD.git
cd SBBD

# Criar ambiente
uv venv --python 3.12 .venv

# Instalar dependências
uv sync --dev

# Validar
uv run pytest -q
# Esperado: 123 passed ✅
```

### Autenticar HuggingFace

```bash
huggingface-cli login
# ou
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

---

## Contribuindo

Contribuições bem-vindas! Áreas:

- Novos adaptadores de engines (Milvus, Qdrant, Weaviate, etc.)
- Datasets adicionais
- Otimizações de performance
- Melhorias de documentação
- Visualizações interativas

---

## Citação

Se você usar BenchVec em sua pesquisa, por favor cite:

```bibtex
@inproceedings{benchvec2026,
  title={BenchVec: Benchmarking Reproduzível de Bancos Vetoriais},
  author={Ferreira, Leandro e Contribuidores},
  booktitle={Simpósio Brasileiro de Bancos de Dados (SBBD)},
  year={2026}
}
```

---

## Licença

Licença MIT — veja arquivo [LICENSE](./LICENSE)

---

## SBBD 2026

Este projeto foi desenvolvido para o **SBBD 2026** (Simpósio Brasileiro de Bancos de Dados).

- 📖 **[README em Inglês](./README.md)**
- 📚 **[Guia de Uso Completo](./docs/COMO_USAR.md)**

Bem-vindo! Welcome! 🎉

---

**Última Atualização:** 29 de março de 2026  
**Status:** ✅ Estável — Pronto para submissão ao SBBD 2026
