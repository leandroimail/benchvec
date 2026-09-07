# BenchVec — Referência Técnica

Referência completa dos engines, workloads, configuração, saídas,
estatísticas e resultados validados do BenchVec. Para uso passo a passo
veja o [Guia de Uso](COMO_USAR.md); para início rápido veja o
[README](../README_pt-BR.md). Versão em inglês: [TECHNICAL.md](TECHNICAL.md).

Toda afirmação aqui é verificável no código de `benchvec/` e na suíte
de testes (`uv run pytest -q` → 305 passed).

---

## 1. O que o BenchVec mede

O BenchVec compara **bancos de dados vetoriais embarcados** no mesmo
hardware, dataset e protocolo, reportando por engine:

| Métrica | Significado |
|---|---|
| `load_time_s` / `build_time_s` | Tempo de ingestão / construção do índice |
| `mean/p95/p99_latency_ms` | Latência de query |
| `qps` | Throughput sustentado de queries |
| `recall_at_k` | Fração dos vizinhos exatos retornados |
| `peak_ram_mb` / `disk_size_mb` | Pico de RAM / pegada persistida |
| `reopen_time_s` | Tempo para reabrir estado persistido (B3) |
| `delete_time_s` / `upsert_time_s` / `post_delete_recall` | Métricas de ciclo de vida (D1) |
| `filter_selectivity` | Fração observada de registros que casam com o filtro (B1) |

Cada execução grava `host_fingerprint` (hardware/software) e linhas
brutas por query (`raw_queries_*.parquet`) — resultados auditáveis até
a medição individual.

## 2. Engines

Todos os adapters implementam o mesmo contrato
(`benchvec/engines/base.py`): `connect / reset / insert / build_index /
search / search_filtered / delete / upsert / persist / reopen /
disk_size / describe_runtime`.

| Engine | Índice | Parâmetros de busca | Métricas | Filtro nativo | Notas |
|---|---|---|---|---|---|
| **faiss** | `flat` ou `ivfflat` (`faiss_nlist`) | `faiss_nprobe` (0 = default da lib) | l2, cosine | Não (pré-filtro na aplicação) | cosine = vetores normalizados + produto interno |
| **chromadb** | HNSW (`ef` fixo na criação) | `search_ef`; grid `ef` opcional (rebuild por ponto) | l2, cosine | Sim (`where`) | grid de ef exige recriar a coleção |
| **lancedb** | `ivfpq` (`num_partitions`, `num_sub_vectors`) | `nprobes`, `refine_factor` | l2, cosine | Sim (`where` SQL) | treino PQ exige ≥ 256 linhas; abaixo disso cai em força bruta |
| **sqlite_vec** | tabela virtual `vec0` (KNN exaustivo) | nenhum (força bruta por design) | l2, cosine | Não (pré-filtro na aplicação) | metadados em tabela SQL companheira (vec0 rejeita NULL) |
| **turbovec** | scan exaustivo TurboQuant (`IdMapIndex`), 2/3/4 bits | `bit_width` (build-time; rebuild por ponto de sweep) | **somente cosine** | Não (filtro na aplicação → allowlist uint64 no kernel) | Rust/PyO3; kernel só produto interno; `dim` múltiplo de 8; calibração TQ+ opcional (~1024 linhas) |

Proteções no config:

- Habilitar `turbovec` com `distance_metric = "l2"` falha **no load do
  config** (o kernel do turbovec não avalia L2).
- O adapter do turbovec mapeia ids string para uint64 densos com
  tombstone no delete (slots nunca são reutilizados ou deslocados),
  mantém metadados em sidecar JSON e retém a amostra de calibração
  para os rebuilds do sweep continuarem calibrados.

## 3. Workloads

Dez workloads (`benchvec/cli.py::DEFAULT_WORKLOADS`). Cada um produz
linhas de `BenchmarkSummary` por engine (por valor de k / nível de
concorrência / tamanho de dataset quando aplicável):

| ID | Nome | O que faz |
|---|---|---|
| `a1` | Ingest | Inserção em lote; mede `load_time_s`, throughput, RAM |
| `a2` | Search | Busca com recall exato em cada `k_values` (k-sweep: ground truth calculado uma vez no k máximo e fatiado por k) |
| `b1` | Filtered search | Busca com filtro de metadados em 4 cenários determinísticos; grava `filter_selectivity` |
| `b2` | Rate-limited ingest | Insert + buscas intercaladas sob teto de taxa de ingestão |
| `b3` | Persistence | `persist()` e mede `reopen_time_s` e busca pós-reabertura |
| `b4` | Reopen | Busca contra estado persistido anteriormente |
| `c1` | Concurrency | Escada de threads (`concurrent_workers`); QPS por wall-clock; query concorrente que falha é retentada sob lock e o nível vira `partial` |
| `c2` | Scaling ladder | Conjunto fixo de queries contra prefixos crescentes do dataset (`sample_sizes`); prefixos por hash estável — todos os engines veem subconjuntos idênticos |
| `d1` | Delete / upsert | Deleta fração determinística (mesma ordenação `:scale` do C2), mede `delete_time_s` e `post_delete_recall` (vs ground truth pré-delete completo), depois faz upsert de registros vivos e "ressuscitados" medindo `upsert_time_s` |
| `sweep` | Parameter sweep | Pontos Recall-QPS sobre grids de parâmetros (ver §5); roda em `k = k_values[0]` |

`a1`–`d1` repetem por `workloads.repetitions` rodadas; `c2` e `sweep`
rodam uma vez por engine (pontos de grid são insensíveis a repetição).

## 4. Ground truth exato

`benchvec/ground_truth.py` calcula vizinhos exatos com a **métrica
configurada** (`engines.distance_metric`):

- `l2`: distância euclidiana ao quadrado (default histórico).
- `cosine`: normaliza L2 ambas as matrizes (zero-safe) e ordena por
  produto interno — ordenação idêntica à distância cosseno; a
  `distance` reportada é `1 − similaridade`.

Ground truth métrico-ciente importa: os embeddings do dataset padrão
**não** são unit-norm (normas ≈ 14,9–19,5 medidas em
`not-lain/wikipedia-small-3000-embedded`) — pontuar hits em ordem
cosine contra ranking L2 enviesaria o recall silenciosamente.

Engines exatos (sqlite_vec, turbovec 4-bit em k baixo) fazendo
`recall = 1.0` sob cosine é a checagem de sanidade embutida de que a
métrica do GT casa com a métrica de busca.

## 5. Parameter sweep (fronteira Recall–QPS)

`run_param_sweep` monta o produto cartesiano do grid de um engine e
mede cada ponto (5 warmups + todas as queries; latência/recall por
ponto gravados nas linhas brutas com `search_params`):

| Engine | Chaves do grid | Aplicação |
|---|---|---|
| faiss | `nprobe` | ao vivo (`set_query_params`) |
| lancedb | `nprobes`, `refine_factor` | ao vivo (atributos) |
| chromadb | `ef` (só se `ef_search_grid_enabled`) | rebuild da coleção + re-ingestão por ponto |
| turbovec | `bit_width` | rebuild do índice quantizado + re-ingestão por ponto (recalibra da amostra retida) |

`max_points` limita o total de pontos medidos; grids acima do orçamento
são subamostrados uniformemente para não amputar a região de alto custo
da fronteira. Engines sem grid (chromadb por padrão, sqlite_vec) são
pulados. O sweep roda em `k = k_values[0]` — no perfil de impacto
(`k_values = [10, 1, 50, 100]`) é k=10, onde as larguras de bits
separam de verdade (em k=1 todas saturam em recall 1.0).

## 6. Estatísticas

`benchvec/statistics.py` (ativado por padrão via `report --stats`):

- **`stats_summary.parquet`**: por grupo `(workload, engine, k,
  concurrency_level, dataset_size)` — n, latência média com IC
  bootstrap semeado, p95, CV entre repetições, recall médio.
- **`stats_pairwise.parquet`**: para cada par de engines no grupo com
  ≥ `MIN_PAIRWISE_SAMPLES` amostras — Mann-Whitney U (assintótico com
  correção de continuidade), delta relativo de média e correção Holm
  step-down entre os pares da família. Linhas de sweep ficam fora dos
  testes pareados (grids heterogêneos) e o agrupamento inclui
  `dataset_size` para não misturar pontos da escada de escala.

## 7. Gráficos

`main.py charts` gera dez PNGs (`benchvec/report.py`):

| Arquivo | Conteúdo |
|---|---|
| `chart_01_latency.png` | latência mean/p95/p99 por engine × workload |
| `chart_02_qps.png` | QPS por engine × workload |
| `chart_03_recall.png` | Recall@K por engine × workload |
| `chart_04_peak_ram.png` | pico de RAM por engine × workload |
| `chart_05_disk_footprint.png` | pegada de disco por engine |
| `chart_06_build_time.png` | build time / time-to-serve |
| `chart_07_concurrency.png` | QPS por nível de concorrência (C1) |
| `chart_08_pareto.png` | fronteira Recall vs QPS por ponto de parâmetros (sweep) |
| `chart_09_scaling.png` | latência média e throughput de ingestão por tamanho de dataset (C2) |
| `chart_10_k_sweep.png` | Recall@K por k (k-sweep do A2) |

O `generate_analysis` ainda renderiza a tabela de significância de
`stats_pairwise.parquet` na análise em markdown/HTML.

## 8. Configuração

Schema completo (`benchvec/config.py`, Pydantic, `extra="forbid"`),
TOML ou YAML. Chaves mais relevantes; defaults valem quando omitidas.

| Seção | Chave | Tipo | Default | Notas |
|---|---|---|---|---|
| — | `project_name` | str | `benchvec` | |
| — | `seed` | int | 42 | Semente de amostragem / ground truth |
| `[paths]` | `data_dir` / `benchmark_dir` / `cache_dir` | path | `.data` / `.benchmarks` / `.cache` | |
| `[dataset]` | `primary` | enum | `fast_wikipedia` | `cohere_wikipedia`, `simple_wikipedia`, `fast_wikipedia`, `sift1m`, `gist1m` |
| | `sample_size` | int | 5000 | Vetores preparados |
| | `ground_truth_k` / `ground_truth_query_count` | int | 100 / 100 | Profundidade e nº de queries do GT |
| `[engines]` | `enabled` | list | `[faiss, chromadb, lancedb]` | + `sqlite_vec`, `turbovec` |
| | `distance_metric` | enum | `l2` | `turbovec` exige `cosine` (falha no load) |
| | `faiss_index_type` / `faiss_nlist` / `faiss_nprobe` | — | `flat` / 100 / 0 | nprobe 0 = default da lib |
| `[engines.lancedb]` | `index_type` / `num_partitions` / `num_sub_vectors` / `nprobes` / `refine_factor` | — | `ivfpq` / 8 / 16 / 8 / 1 | refine_factor ≥ 2 ativa refinamento |
| `[engines.chroma]` | `search_ef` / `ef_search_grid_enabled` / `ef_search_grid` | — | 64 / false / [] | grid reconstrói por ponto |
| `[engines.turbovec]` | `bit_width` / `calibrate` | 2·3·4 / bool | 4 / false | knob de build |
| `[workloads]` | `repetitions` / `warmup_queries` / `k_values` / `ingest_batch_sizes` / `ingest_rates_per_second` / `concurrent_workers` / `sample_sizes` / `max_c2_queries` | — | 3 / 10 / [10] / [1000] / [100] / [1] / [] / 20000 | `k_values[0]` comanda o sweep |
| `[workloads.param_sweep]` | `faiss_nprobe` / `lancedb_nprobes` / `lancedb_refine_factor` / `turbovec_bit_width` / `max_points` | list[int] / int | [] / [] / [] / [] / 20000 | grids; `turbovec_bit_width ∈ {2,3,4}` |
| `[timeouts]` | segundos por fase | int | 1800/1800/300/900/600 | resultados parciais capturados no timeout |

## 9. CLI

`uv run python main.py <comando>` (entrypoint `main.py`):

| Comando | Propósito |
|---|---|
| `run` | Workloads selecionados (`--workload` repetível) para os engines habilitados; `--config`, `--project` (default `benchmark_default`), `--output-dir`, `--keep-engine-storage` |
| `run-all` | Suíte completa padrão (os 10 workloads) |
| `report` | Consolida os summaries em um parquet (`--input`/`--project`, `--output`; `--stats/--no-stats`, padrão ligado) |
| `charts` | Gera os 10 PNGs (`--input`, `--output`) |
| `analyze` | Análise em Markdown/HTML/JSON (`--format`, `--report-type`) |
| `inspect` | Inspeciona um summary ou os records brutos (`--project`, `--run-id`, `--summary`, `--input`) |
| `init` | Cria a pasta do projeto (`--project`, `--template <config.toml>`) |
| `download` | Baixa e cacheia o dataset configurado |

Flags úteis:

- `--keep-engine-storage` retém o storage por run (padrão: apagar após
  capturar métricas — higiene de disco; o storage compartilhado do C1 é
  limpo pelo CLI após a invocação).
- Passe sempre `--config configs/impact-benchmark.toml` explicitamente
  para o perfil completo.

## 10. Projetos e saídas

```
.benchmarks/projects/<projeto>/
├── project.toml            # metadados do projeto
├── config.toml             # cópia opcional de config (init --template)
├── results/<engine>/<workload>/<run_id>/
│   ├── summary.parquet     # uma linha de BenchmarkSummary
│   ├── env.json            # fingerprint do host + versões
│   └── raw_queries_*.parquet / raw_ingest_*.parquet
├── reports/                # parquet consolidado, stats, charts/, análise
└── logs/
```

`benchmark_default` é o projeto padrão quando `--project` é omitido.
Nota: `.benchmarks/` está no gitignore — artefatos são reprodutíveis,
não arquivados.

## 11. Perfis prontos

| Config | Propósito |
|---|---|
| `configs/first-benchmark.toml` | Benchmark oficial completo (Cohere Wikipedia) |
| `configs/fast-benchmark.toml` | Perfil rápido de smoke test |
| `configs/impact-benchmark.toml` | **Campanha completa, 10 workloads, 5 engines, cosine** — o perfil principal |

Perfil de impacto, medido num MacBook M5 (16 GB, cache de dataset
morno):

- 233 summaries, **todos `status=success`**; ~8,8 min de parede
  (execução limpa); 9,4 MB em disco com a limpeza padrão.
- Engines: faiss (IVFFlat nprobe 1–32), lancedb (IVF_PQ nprobes 1–16),
  chromadb (ponto único de ef), sqlite_vec (sem grid), turbovec (grid
  de rebuild 2/3/4 bits).
- Sweep de bit-width do TurboVec em k=10: recall 0,899 / 0,956 / 0,973
  a ~0,09 ms — spread de Pareto genuíno (em k=1 todas saturam em 1.0;
  por isso `k_values = [10, 1, 50, 100]`).
- C1 do TurboVec escala nativamente (≈3,3–4,5k QPS @1 thread → 6,3–7,4k
  @8 threads; sem resultados `partial` por lock).
- Quirk medido (turbovec 1.0.0): snapshots de índice 3-bit e 4-bit têm
  o mesmo tamanho (packing interno de 4 bits); só o 2-bit comprime
  (~0,53× do 4-bit).
- Âncora de sanidade: sqlite_vec (força bruta exata) faz recall 1.0 em
  todo k sob cosine, confirmando o alinhamento GT/métrica de busca em
  toda a campanha.

## 12. Decisões de design e limitações

- **Ground truth métrico-ciente**: o GT honra
  `engines.distance_metric`; o perfil de impacto usa cosine porque o
  turbovec é só produto interno e engines só são comparáveis dentro de
  uma métrica.
- **Mapeamento de ids do TurboVec**: atribuição densa uint64 com
  tombstone no delete; sidecars `ids.json`/`metadata.json`; snapshot
  sem sidecar de ids utilizável é descartado no reopen (estado limpo)
  em vez de divergir.
- **Documentos não são armazenados no TurboVec** (hits trazem id +
  score); metadados ficam só para avaliar filtros. As métricas nunca
  leem payload além dos ids.
- **chroma `ef` e turbovec `bit_width` são knobs de build**: pontos de
  sweep reconstruem o índice e re-ingestionam (medido dentro do sweep).
- **Deletes do faiss são tombstones**: espaço não é recuperado até um
  rebuild; o D1 mede o comportamento como é.
- **Piso de treino PQ do lancedb**: < 256 linhas cai silenciosamente em
  força bruta; o adapter avisa e pula IVF_PQ em escalas mínimas.
- **C1**: engines sem clientes thread-safe expõem níveis `partial`
  honestos via retentativa com lock (na prática turbovec, faiss, chroma,
  lancedb e sqlite_vec rodam nativos).
- **Timeouts** capturam resultados parciais por fase
  (`benchvec/timeouts.py`) em vez de falhar o run.

## 13. Testes

```bash
uv run pytest -q          # 305 passed
```

Cobrem validação de config, todos os adapters (incluindo turbovec:
round-trips, tombstones, filtros, persistência, calibração, rebuild,
concorrência), métricas de ground truth, orquestração do runner,
comandos de CLI, relatórios/estatísticas e o gerador de análise. Testes
pulam de forma limpa quando uma dependência nativa não está disponível
no host.
