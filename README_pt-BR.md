# BenchVec: Suite de Benchmarking Reproduzível de Bancos de Dados Vetoriais

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com)
[![Tests](https://img.shields.io/badge/tests-305%20passed-blue)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

BenchVec é uma suite Python para **comparação justa e reproduzível** de bancos de dados vetoriais embarcados (FAISS, ChromaDB, LanceDB, sqlite-vec, TurboVec) com metodologia científica rigorosa.

**[English](./README.md) | [Documentação](#documentação) | [Quick Start](#quick-start)**

---

## Visão Geral

### Problema

Escolher entre engines de bancos vetoriais para aplicações locais de IA (RAG, assistentes, bases de conhecimento) exige benchmarking rigoroso e reprodutível:

- **Variância de performance** entre hardwares, configurações e workloads
- **Comparações injustas** entre bibliotecas de indexação (FAISS) e bancos completos (ChromaDB, LanceDB)
- **Falta de rastreabilidade** — sem registro das medições individuais ou anomalias
- **Desafios de reprodutibilidade** — reprodutibilidade por semente é crítica para trabalhos científicos

### Solução

O BenchVec oferece:

✅ **Design de benchmark metodologicamente sólido:**
  - Workloads separados para busca ANN pura e busca filtrada com metadados
  - Fingerprint completo de hardware (CPU, RAM, tipo de storage)
  - Timeouts, warm-up e repetições configuráveis
  - Validação por ground truth para as métricas de Recall@K

✅ **Dados de medição ricos:**
  - Métricas agregadas (latência, QPS, recall, memória, disco)
  - **Micro-dados brutos** de cada query e lote de ingestão
  - Rastreabilidade total via armazenamento em Parquet
  - Captura do ambiente (OS, versão do Python, versões das bibliotecas)

✅ **Relatórios automatizados:**
  - 10 gráficos profissionais de comparação (PNG)
  - Exportação em Parquet/CSV para análise
  - Inspeção por run via CLI
  - Consolidação entre múltiplas execuções

✅ **Reprodutibilidade:**
  - Amostragem determinística do dataset (com controle de seed)
  - Embeddings pré-computados (Cohere Wikipedia com dimensões fixas)
  - Log exato da configuração
  - Configs de exemplo para cenários comuns

---

## Funcionalidades Principais

| Funcionalidade | Benefício |
|---------|---------|
| **Cobertura completa de workloads** | A1-B4: ingestão, busca pura, busca filtrada, persistência, reabertura de estado; C1: busca concorrente (escada de threads); C2: escada de escala de dados; D1: ciclo de delete/upsert; Sweep: varredura Recall-QPS |
| **Datasets oficiais** | SIFT1M, GIST1M, Cohere Wikipedia 22-12 com embeddings pré-computados |
| **Metadados-consciente** | Metadados sintéticos mas determinísticos para cenários RAG realistas |
| **Políticas de timeout** | Timeouts configuráveis por fase com captura de resultados parciais |
| **Multi-engine** | FAISS, ChromaDB, LanceDB, sqlite-vec (força bruta), TurboVec (quantização TurboQuant) com arquitetura plugável |
| **Índices ANN em todo lugar** | FAISS IVFFlat (nprobe), ChromaDB HNSW (ef_search), LanceDB IVF_PQ, TurboVec TurboQuant (sweep 2/3/4 bits) — sem comparações scan-only escondidas |
| **Testes concorrentes** | C1 mede QPS e latência sob 1-8 threads |
| **Rigor estatístico** | ICs bootstrap, CV entre repetições, Mann-Whitney pareado com correção de Holm |
| **Organização de projetos** | Benchmarks isolados com layouts de diretório estruturados |
| **Relatórios visuais** | 10 gráficos automáticos de comparação (latência, QPS, recall, RAM, disco, build time, concorrência, fronteira de Pareto, escalabilidade, k-sweep) |
| **Higiene de disco** | Armazenamento dos engines limpo após a captura das métricas (`--keep-engine-storage` para reter) |

---

## Quick Start

### Instalação

```bash
git clone git@github.com:leandroimail/benchvec.git
cd benchvec

# Python 3.10+ com uv
uv sync --dev

# Valide
uv run pytest -q
# Esperado: 305 passed
```

### Primeiro Benchmark

```bash
# Autentique no HuggingFace (para o dataset Cohere Wikipedia)
huggingface-cli login

# Rode o benchmark completo
uv run python main.py run-all --config configs/first-benchmark.toml

# Veja os resultados
RUNID=$(ls .benchmarks/projects/benchmark_default/results/*/a1/run-* | head -1 | xargs basename)
uv run python main.py inspect --run-id $RUNID --summary
```

### Perfil Rápido (<5 minutos)

Para validação rápida e iteração de desenvolvimento, use o config fast benchmark com até 5.000 registros e batch de 1.000:

```bash
# Benchmark rápido (sem autenticação HuggingFace — usa dataset público)
uv run python main.py run-all --config configs/fast-benchmark.toml

# Inicialize projeto com perfil rápido
uv run python main.py init --project fast-test --template configs/fast-benchmark.toml
uv run python main.py run-all --project fast-test
```

**Dataset:** `not-lain/wikipedia-small-3000-embedded` — 3.000 registros da Wikipedia com embeddings pré-computados (público, sem autenticação).

| Parâmetro | Perfil Completo | Teste Rápido |
|-----------|-------------|-----------|
| sample_size | 180 | **5.000** |
| ingest_batch_sizes | [60] | **[1.000]** |
| repetitions | 10 | **3** |
| warmup_queries | 20 | **10** |
| Tempo alvo | Minutos | **<5 min** |

### Perfil de Impacto (Campanha Completa)

Todos os workloads de expansão — concorrência (C1), escalabilidade (C2),
delete/upsert (D1), parameter sweep, k-sweep — com índices ANN em todos
os engines:

```bash
uv run python main.py init --project impact --template configs/impact-benchmark.toml
uv run python main.py run-all --project impact
uv run python main.py report --project impact --output .benchmarks/projects/impact/reports/final.parquet
uv run python main.py charts --project impact
uv run python main.py analyze --project impact --output .benchmarks/projects/impact/reports/analysis.md
```

Medido num MacBook M5 (16 GB) com a campanha cosine de 5 engines
(faiss, chromadb, lancedb, sqlite_vec, turbovec): 233 summaries, todos
`success`, ~8,8 min de parede (execução limpa), 9,4 MB em disco com a
limpeza padrão de storage. O sweep de bit-width do TurboVec distribui
recall@10 entre 0,899 / 0,956 / 0,973 em 2/3/4 bits. Veja
`configs/impact-benchmark.toml` para o orçamento medido e a
[Referência Técnica](docs/TECHNICAL_pt-BR.md) para a motivação do design.

### Gerar Relatórios de Análise

```bash
# Relatório em markdown a partir da execução completa
uv run python main.py analyze --project benchmark_default --output analysis.md

# Análise JSON para processamento programático
uv run python main.py analyze --project benchmark_default --format json --output analysis.json

# Relatório HTML com estatísticas detalhadas
uv run python main.py analyze --project benchmark_default --format html --output analysis.html
```

### Gerar Gráficos de Comparação

```bash
uv run python main.py charts --project benchmark_default

# Abra os resultados
open .benchmarks/projects/benchmark_default/reports/charts/
```

**Saídas esperadas:**
- `chart_01_latency.png` — Latência (mean/p95/p99) por engine
- `chart_02_qps.png` — Throughput (queries/segundo)
- `chart_03_recall.png` — Recall@K (alvo: 1.0)
- `chart_04_peak_ram.png` — Pico de memória
- `chart_05_disk_footprint.png` — Armazenamento em disco por engine
- `chart_06_build_time.png` — Construção de índice + time-to-serve
- `chart_07_concurrency.png` — QPS + latência p95 por nº de threads (C1)
- `chart_08_pareto.png` — Fronteira Recall vs QPS por ponto de parâmetros (Sweep)
- `chart_09_scaling.png` — Latência + throughput de ingestão por tamanho de dataset (C2)
- `chart_10_k_sweep.png` — Recall e latência por k (k-sweep do A2)

---

## Documentação

### Referência Completa

- **[TECHNICAL_pt-BR.md](docs/TECHNICAL_pt-BR.md)** — Referência técnica completa (português)
  - Matriz de capacidades dos engines (faiss, chromadb, lancedb, sqlite_vec, turbovec)
  - Os 10 workloads e o protocolo de ground truth exato
  - Schema de configuração, sweeps de parâmetros, estatísticas, gráficos
  - Resultados validados da campanha e decisões de design/limitações

### Para Usuários

- **[USER_GUIDE.md](docs/USER_GUIDE.md)** — Guia completo em inglês
  - Instalação, configuração, todos os comandos CLI
  - Descrição de datasets (SIFT1M, Cohere Wikipedia, Wikipedia local)
  - Explicação dos workloads
  - Troubleshooting
  - Boas práticas de benchmarking

- **[COMO_USAR.md](docs/COMO_USAR.md)** — Guia completo em português
  - Instalação, configuração, todos os comandos CLI
  - Descrição de datasets
  - Explicação dos workloads
  - Troubleshooting

### Para Pesquisadores

- **Referência técnica completa** → veja [Referência Técnica](docs/TECHNICAL_pt-BR.md) (português) ou [Technical Reference](docs/TECHNICAL.md) (inglês)
- **Benchmark completo** → veja [Benchmark Principal](docs/COMO_USAR.md#benchmark-principal) (português) ou [Main Benchmark](docs/USER_GUIDE.md#main-benchmark) (inglês)
- **Datasets oficiais** → veja [Datasets Oficiais](docs/COMO_USAR.md#datasets-oficiais) (português) ou [Official Datasets](docs/USER_GUIDE.md#official-datasets) (inglês)
- **Definições dos workloads** → seção [Workloads](docs/USER_GUIDE.md#workloads) no USER_GUIDE.md
- **Guia de interpretação** → veja [Gerando Gráficos](docs/COMO_USAR.md#gerando-gráficos) (português) ou [Generating Charts](docs/USER_GUIDE.md#generating-charts) (inglês)

---

## Estrutura do Projeto

```
benchvec/
├── README.md                     # Este arquivo (inglês)
├── README_pt-BR.md              # README em português
├── benchvec/                    # Pacote principal
│   ├── cli.py                   # Comandos CLI
│   ├── config.py                # Configuração
│   ├── datasets.py              # Carregamento de datasets
│   ├── chunker.py               # Segmentação de texto
│   ├── ground_truth.py          # Ground truth
│   ├── metrics.py               # Medições
│   ├── runner.py                # Orquestração dos benchmarks
│   ├── raw_store.py             # Armazenamento de micro-dados
│   ├── report.py                # Relatórios e gráficos
│   ├── host.py                  # Fingerprint de hardware
│   ├── statistics.py            # IC bootstrap, Mann-Whitney U + Holm, CV
│   ├── project.py               # Gerenciamento de projetos
│   └── engines/                 # Adaptadores de engines
│       ├── base.py
│       ├── faiss_engine.py
│       ├── chroma_engine.py
│       ├── lancedb_engine.py
│       ├── sqlite_vec_engine.py # baseline sqlite-vec (força bruta)
│       └── turbovec_engine.py   # índice quantizado TurboVec (só cosine)
├── configs/                     # Configurações de exemplo
│   ├── first-benchmark.toml     # Benchmark oficial
│   ├── fast-benchmark.toml      # Perfil rápido (<5 min)
│   └── impact-benchmark.toml    # Perfil completo (C1/C2/D1/sweep)
├── docs/
│   ├── USER_GUIDE.md           # Guia passo a passo (inglês)
│   ├── TECHNICAL.md            # Referência técnica completa (inglês)
│   ├── TECHNICAL_pt-BR.md      # Referência técnica completa (português)
│   └── COMO_USAR.md            # Guia do usuário (português)
├── tests/
│   ├── unit/                    # 305 testes unitários
│   └── conftest.py
└── .benchmarks/                 # Resultados (criado automaticamente)
```

---

## Requisitos

- **Python:** 3.10+
- **OS:** Linux, macOS, Windows (WSL2)
- **RAM:** 4+ GB (8+ GB recomendado para SIFT1M completo)
- **Disco:** 10+ GB (cache e resultados)
- **HuggingFace:** Conta + autenticação (para Cohere Wikipedia)

---

## Instalação

```bash
# Clone o repositório
git clone git@github.com:leandroimail/benchvec.git
cd benchvec

# Crie o ambiente
uv venv --python 3.12 .venv

# Instale as dependências
uv sync --dev

# Valide
uv run pytest -q
# Esperado: 305 passed ✅
```

### Autenticação HuggingFace

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

Se você usar BenchVec em sua pesquisa, por favor cite o projeto:

```bibtex
@misc{benchvec2026,
  title={BenchVec: Benchmarking Reproduzível de Bancos Vetoriais},
  author={Ferreira, Leandro e Contribuidores},
  year={2026},
  howpublished={\url{https://github.com/leandroimail/benchvec}}
}
```

---

## Licença

Licença MIT — veja o arquivo [LICENSE](./LICENSE)

---

**Última atualização:** 7 de setembro de 2026  
**Status:** ✅ Estável

## Uso da CLI

### Modo Projeto

```bash
# Inicialize o projeto
uv run python main.py init --project meu-projeto

# Rode benchmarks no projeto
uv run python main.py run --project meu-projeto --config config.toml
uv run python main.py run-all --project meu-projeto

# Relatórios e inspeção
uv run python main.py report --project meu-projeto
uv run python main.py charts --project meu-projeto
uv run python main.py inspect --project meu-projeto --run-id run-123
```

### Modo Legado (Sem Projetos)

```bash
uv run python main.py run --config config.toml
uv run python main.py run-all --config config.toml
uv run python main.py report --input .benchmarks/results --output reports/final.parquet
uv run python main.py inspect --run-id run-123 --summary
uv run python main.py inspect --run-id run-123 --type queries
uv run python main.py inspect --run-id run-123 --type ingest
```

## Cobertura Atual

A implementação atual inclui:

- carregamento de configuração em YAML e TOML
- coleta de fingerprint de hardware e helpers de timeout
- cache de dataset preparado para Cohere Wikipedia com fallback Simple Wikipedia
- geração determinística de metadados sintéticos e chunking
- cálculo e cache de ground truth exato métrico-ciente
- adaptadores de engine para FAISS (flat/IVFFlat), ChromaDB (HNSW), LanceDB (IVF_PQ), sqlite-vec (força bruta) e TurboVec (scan exaustivo TurboQuant; só cosine, larguras 2/3/4 bits, calibração TQ+ opcional)
- métodos de runner para A1-A2, B1-B4, mais C1 (busca concorrente), C2 (escada de escala), D1 (ciclo delete/upsert) e o parameter sweep Recall-QPS
- controle de parâmetros de busca por engine: FAISS `nprobe`, LanceDB `nprobes`/`refine_factor` (índice IVF_PQ construído automaticamente), ChromaDB `ef_search`, TurboVec `bit_width` (rebuild por ponto de grid)
- ground truth exato métrico-ciente: o GT honra `engines.distance_metric` (turbovec exige `cosine`, o kernel é só produto interno; o validador fail-fast rejeita turbovec + l2 no load)
- k-sweep: ground truth calculado uma vez no k máximo e fatiado por k
- módulo de estatísticas: ICs bootstrap, CV entre repetições, Mann-Whitney U pareado com correção Holm step-down (`stats_summary.parquet` / `stats_pairwise.parquet`)
- 10 gráficos comparativos incluindo a fronteira de Pareto Recall-QPS e escalabilidade de concorrência
- persistência de summaries, micro-dados brutos, inspeção de runs e consolidação de relatórios
- **organização de artefatos por projeto com estrutura de diretórios automática**
- higiene de disco: storage por run é limpo após capturar métricas (`--keep-engine-storage` para reter)
- comandos CLI para `init`, `run`, `run-all`, `report --stats`, `inspect`, `charts` e `analyze` com flag opcional `--project`

## Notas

- Os artefatos de benchmark são escritos em `.benchmarks/projects/{project_name}/` no modo projeto, ou `.benchmarks/results` no modo legado.
- O cache do dataset preparado fica em `.cache/<nome-do-dataset>`.
- O acesso ao Cohere Wikipedia depende das permissões necessárias no Hugging Face.
- Os arquivos da raiz `AGENTS.md` e `main.py` ficam fora dos commits de implementação, salvo pedido explícito.
