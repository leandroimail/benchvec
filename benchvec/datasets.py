"""Dataset loading, deterministic sampling, metadata synthesis and local caching."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tarfile
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from benchvec.chunker import (
    TokenChunkerProtocol,
    build_token_chunker,
    chunk_document,
    chunker_to_metadata,
)
from benchvec.config import DatasetConfig

DATASET_FAST_SAMPLE_THRESHOLD = 5000

DEFAULT_AUTHORS = [
    "engenharia",
    "pesquisa",
    "produto",
    "operacoes",
    "plataforma",
    "arquitetura",
    "dados",
    "ia",
]

FALLBACK_CATEGORIES = [
    "data_engineering",
    "artificial_intelligence",
    "software_engineering",
    "science",
    "humanities",
    "general",
]

CATEGORY_RULES = [
    (
        "data_engineering",
        [
            "database",
            "sql",
            "etl",
            "pipeline",
            "data warehouse",
            "data lake",
            "apache spark",
            "hadoop",
        ],
    ),
    (
        "artificial_intelligence",
        [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural network",
            "language model",
            "nlp",
            "computer vision",
        ],
    ),
    (
        "software_engineering",
        [
            "software",
            "programming",
            "algorithm",
            "compiler",
            "operating system",
            "internet",
            "protocol",
        ],
    ),
    (
        "science",
        [
            "biology",
            "chemistry",
            "physics",
            "medicine",
            "disease",
            "astronomy",
            "mathematics",
        ],
    ),
    (
        "humanities",
        [
            "history",
            "politics",
            "philosophy",
            "religion",
            "war",
            "literature",
            "economics",
        ],
    ),
]

TAG_POOLS = {
    "data_engineering": ["engenharia_dados", "etl", "sql", "analytics"],
    "artificial_intelligence": ["ia", "ml", "nlp", "rag"],
    "software_engineering": ["software", "arquitetura", "programacao", "sistemas"],
    "science": ["ciencia", "pesquisa", "metodo", "experimento"],
    "humanities": ["historia", "sociedade", "cultura", "politica"],
    "general": ["geral", "referencia", "enciclopedia", "conhecimento"],
}

DATE_START = dt.date(2021, 1, 1)
DATE_END = dt.date(2025, 12, 31)
DATE_RANGE_DAYS = (DATE_END - DATE_START).days + 1

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


class PreparedRecord(BaseModel):
    """Canonical prepared record used by benchmark workloads."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    embedding: list[float]
    source_dataset: str
    document_title: str
    document_length_bucket: str
    categoria: str
    tag: str
    autor: str
    data_criacao: int
    parent_document_id: str | None = None
    token_count: int | None = Field(default=None, ge=0)


class PreparedDatasetManifest(BaseModel):
    """Persistent metadata for a prepared local dataset cache."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_variant: str
    sample_size: int
    seed: int
    chunking: dict[str, Any] | None = None
    selected_ids_file: str = "selected_ids.json"
    records_file: str = "records.parquet"
    doc_chunk_map_file: str = "doc_chunk_map.json"


def stable_hash(text: str) -> int:
    """Generate a deterministic integer hash for reproducible sampling."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def canonical_text(title: str, text: str) -> str:
    """Build the canonical textual representation mandated by PRD03."""

    clean_title = title.strip()
    clean_text = text.strip()
    if clean_title and clean_text:
        return f"{clean_title}\n\n{clean_text}"
    return clean_title or clean_text


def token_count(text: str) -> int:
    """Count whitespace-delimited tokens for metadata bucketing."""

    return len(text.split())


def length_bucket(text: str) -> str:
    """Classify text length according to the PRD03 buckets."""

    count = token_count(text)
    if count < 128:
        return "short"
    if count < 512:
        return "medium"
    if count < 1024:
        return "long"
    return "xlong"


def infer_category(record_id: str, seed: int, title: str, text: str) -> str:
    """Classify the synthetic category deterministically."""

    content = f"{title} {text}".lower()
    for category, keywords in CATEGORY_RULES:
        if any(keyword in content for keyword in keywords):
            return category
    index = stable_hash(f"{seed}:{record_id}:category") % len(FALLBACK_CATEGORIES)
    return FALLBACK_CATEGORIES[index]


def infer_tag(record_id: str, seed: int, category: str) -> str:
    """Choose a deterministic synthetic tag from the category pool."""

    pool = TAG_POOLS[category]
    return pool[stable_hash(f"{seed}:{record_id}:{category}:tag") % len(pool)]


def infer_author(record_id: str, seed: int) -> str:
    """Choose a deterministic synthetic author."""

    return DEFAULT_AUTHORS[stable_hash(f"{seed}:{record_id}:author") % len(DEFAULT_AUTHORS)]


def infer_date(record_id: str, seed: int) -> int:
    """Generate a deterministic synthetic creation timestamp (UTC epoch seconds)."""

    offset = stable_hash(f"{seed}:{record_id}:date") % DATE_RANGE_DAYS
    synthetic_date = DATE_START + dt.timedelta(days=offset)
    return int(dt.datetime.combine(synthetic_date, dt.time.min, dt.timezone.utc).timestamp())


def is_valid_wikipedia_record(record: dict[str, Any], min_chars: int) -> bool:
    """Filter empty, redirect and too-short raw Wikipedia records."""

    text = str(record.get("text") or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("#redirect") or lowered.startswith("redirect "):
        return False
    return len(text) >= min_chars


def deterministic_sample_ids(record_ids: Sequence[str], sample_size: int, seed: int) -> list[str]:
    """Select a deterministic subset of IDs sorted by stable seeded rank."""

    ranked_ids = sorted(
        record_ids,
        key=lambda record_id: stable_hash(f"{seed}:{record_id}:sample"),
    )
    return ranked_ids[:sample_size]


def _extract_embedding(raw_record: dict[str, Any]) -> list[float]:
    embedding = raw_record.get("embedding")
    if embedding is None:
        embedding = raw_record.get("emb")
    if embedding is None:
        embedding = raw_record.get("embeddings")
    if embedding is None:
        raise ValueError("Record is missing an embedding field")
    return [float(value) for value in cast(Sequence[float], embedding)]


def _metadata_fields(record_id: str, title: str, text: str, seed: int) -> dict[str, str | int]:
    category = infer_category(record_id, seed, title, text)
    canonical = canonical_text(title, text)
    return {
        "document_title": title,
        "document_length_bucket": length_bucket(canonical),
        "categoria": category,
        "tag": infer_tag(record_id, seed, category),
        "autor": infer_author(record_id, seed),
        "data_criacao": infer_date(record_id, seed),
    }


def prepare_cohere_wikipedia_records(
    raw_records: Iterable[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[PreparedRecord]:
    """Prepare pre-embedded Cohere Wikipedia passages with synthetic metadata."""

    indexed_records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        record_id = str(
            raw_record.get("id") or raw_record.get("_id") or raw_record.get("uuid") or ""
        )
        if not record_id:
            continue
        indexed_records[record_id] = raw_record

    selected_ids = deterministic_sample_ids(
        list(indexed_records),
        sample_size=sample_size,
        seed=seed,
    )
    prepared_records: list[PreparedRecord] = []
    for record_id in selected_ids:
        raw_record = indexed_records[record_id]
        title = str(raw_record.get("title") or raw_record.get("document_title") or "").strip()
        text = str(
            raw_record.get("text") or raw_record.get("passage") or raw_record.get("content") or ""
        ).strip()
        metadata = _metadata_fields(record_id, title, text, seed)
        prepared_records.append(
            PreparedRecord(
                id=record_id,
                text=text or canonical_text(title, text),
                embedding=_extract_embedding(raw_record),
                source_dataset="cohere_wikipedia",
                document_title=str(metadata["document_title"]),
                document_length_bucket=str(metadata["document_length_bucket"]),
                categoria=str(metadata["categoria"]),
                tag=str(metadata["tag"]),
                autor=str(metadata["autor"]),
                data_criacao=int(metadata["data_criacao"]),
            )
        )
    return prepared_records


def _prepare_fast_wikipedia_records(
    raw_records: Iterable[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[PreparedRecord]:
    prepared_records: list[PreparedRecord] = []
    for raw_record in raw_records:
        record_id = str(
            raw_record.get("id")
            or raw_record.get("_id")
            or raw_record.get("uuid")
            or stable_hash(json.dumps(raw_record, sort_keys=True))
        )
        title = str(raw_record.get("title") or "").strip()
        text = str(raw_record.get("text") or raw_record.get("passage") or "").strip()
        if not text:
            continue
        metadata = _metadata_fields(record_id, title, text, seed)
        prepared_records.append(
            PreparedRecord(
                id=record_id,
                text=text or canonical_text(title, text),
                embedding=_extract_embedding(raw_record),
                source_dataset="fast_wikipedia",
                document_title=str(metadata["document_title"]),
                document_length_bucket=str(metadata["document_length_bucket"]),
                categoria=str(metadata["categoria"]),
                tag=str(metadata["tag"]),
                autor=str(metadata["autor"]),
                data_criacao=int(metadata["data_criacao"]),
            )
        )
        if len(prepared_records) >= sample_size:
            break
    return prepared_records


def prepare_simple_wikipedia_records(
    raw_records: Iterable[dict[str, Any]],
    config: DatasetConfig,
    embedding_function: EmbeddingFunction,
    *,
    chunker: TokenChunkerProtocol | None = None,
) -> list[PreparedRecord]:
    """Prepare raw Simple Wikipedia documents through chunking and local embedding."""

    indexed_records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not is_valid_wikipedia_record(raw_record, config.min_text_characters):
            continue
        record_id = str(
            raw_record.get("id")
            or raw_record.get("_id")
            or stable_hash(json.dumps(raw_record, sort_keys=True))
        )
        indexed_records[record_id] = raw_record

    selected_ids = deterministic_sample_ids(
        list(indexed_records),
        sample_size=config.sample_size,
        seed=config.seed,
    )
    token_chunker = chunker or build_token_chunker(
        tokenizer_name=config.local_embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    prepared_rows: list[dict[str, Any]] = []
    for record_id in selected_ids:
        raw_record = indexed_records[record_id]
        title = str(raw_record.get("title") or "").strip()
        text = str(raw_record.get("text") or "").strip()
        canonical = canonical_text(title, text)
        metadata = _metadata_fields(record_id, title, text, config.seed)
        for chunk in chunk_document(record_id, canonical, token_chunker):
            prepared_rows.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_dataset": "simple_wikipedia",
                    "document_title": metadata["document_title"],
                    "document_length_bucket": metadata["document_length_bucket"],
                    "categoria": metadata["categoria"],
                    "tag": metadata["tag"],
                    "autor": metadata["autor"],
                    "data_criacao": metadata["data_criacao"],
                    "parent_document_id": record_id,
                    "token_count": chunk.token_count,
                }
            )

    embeddings = embedding_function([row["text"] for row in prepared_rows])
    prepared_records: list[PreparedRecord] = []
    for row, embedding in zip(prepared_rows, embeddings):
        prepared_records.append(
            PreparedRecord(
                id=str(row["id"]),
                text=str(row["text"]),
                embedding=[float(value) for value in embedding],
                source_dataset=str(row["source_dataset"]),
                document_title=str(row["document_title"]),
                document_length_bucket=str(row["document_length_bucket"]),
                categoria=str(row["categoria"]),
                tag=str(row["tag"]),
                autor=str(row["autor"]),
                data_criacao=int(row["data_criacao"]),
                parent_document_id=str(row["parent_document_id"]),
                token_count=int(row["token_count"]),
            )
        )
    return prepared_records


def sentence_transformer_embedder(model_name: str) -> EmbeddingFunction:
    """Create a local embedding function based on sentence-transformers."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def _embed(texts: list[str]) -> list[list[float]]:
        encoded = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [[float(value) for value in row] for row in encoded.tolist()]

    return _embed


def load_huggingface_dataset_records(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    *,
    streaming: bool,
    max_records: int | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
) -> Iterable[dict[str, Any]]:
    """Load a Hugging Face dataset split with optional auth token."""

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    records: Iterable[dict[str, Any]] = cast(Iterable[dict[str, Any]], dataset)
    if streaming and max_records is not None:
        records = cast(Iterable[dict[str, Any]], islice(records, max_records))
    return records


def persist_prepared_dataset(
    output_dir: str | Path,
    records: Sequence[PreparedRecord],
    manifest: PreparedDatasetManifest,
) -> None:
    """Persist prepared benchmark records and reproducibility sidecars locally."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    records_path = target_dir / manifest.records_file
    selected_ids_path = target_dir / manifest.selected_ids_file
    doc_chunk_map_path = target_dir / manifest.doc_chunk_map_file
    manifest_path = target_dir / "manifest.json"

    table = pa.Table.from_pylist([record.model_dump(mode="json") for record in records])
    pq.write_table(table, records_path)

    selected_ids = [record.parent_document_id or record.id for record in records]
    selected_ids_path.write_text(json.dumps(sorted(set(selected_ids)), indent=2), encoding="utf-8")

    doc_chunk_map: dict[str, list[str]] = {}
    for record in records:
        parent_document_id = record.parent_document_id or record.id
        doc_chunk_map.setdefault(parent_document_id, []).append(record.id)
    doc_chunk_map_path.write_text(json.dumps(doc_chunk_map, indent=2), encoding="utf-8")

    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def load_prepared_dataset(
    output_dir: str | Path,
) -> tuple[PreparedDatasetManifest, list[PreparedRecord]]:
    """Load a previously persisted prepared dataset cache."""

    target_dir = Path(output_dir)
    manifest = PreparedDatasetManifest.model_validate_json(
        (target_dir / "manifest.json").read_text(encoding="utf-8")
    )
    table = pq.read_table(target_dir / manifest.records_file)
    records = [
        PreparedRecord.model_validate(record)
        for record in cast(list[dict[str, Any]], table.to_pylist())
    ]
    return manifest, records


def _read_fvecs(path: Path) -> np.ndarray:
    """Read TexMex .fvecs vectors into a float32 matrix."""

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    if np.any(rows[:, 0] != dim):
        msg = f"Invalid fvecs format: {path}"
        raise ValueError(msg)
    return rows[:, 1:].view(np.float32)


def _texmex_file_map(dataset: str) -> tuple[str, str]:
    """Return tarball and base vector filename for a TexMex dataset."""

    if dataset == "sift1m":
        return "sift.tar.gz", "sift_base.fvecs"
    if dataset == "gist1m":
        return "gist.tar.gz", "gist_base.fvecs"
    msg = f"Unsupported TexMex dataset: {dataset}"
    raise ValueError(msg)


def _ensure_texmex_dataset(config: DatasetConfig, dataset: str, target_dir: Path) -> Path:
    """Ensure TexMex dataset exists locally, downloading and extracting if needed."""

    tar_name, base_file = _texmex_file_map(dataset)
    base_path = target_dir / base_file
    if base_path.exists():
        return base_path

    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / tar_name
    if not tar_path.exists():
        url = f"{config.texmex_base_url.rstrip('/')}/{tar_name}"
        url_timeout = config.texmex_download_timeout_seconds
        with urllib.request.urlopen(url, timeout=url_timeout) as response:
            tar_path.write_bytes(response.read())

    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(target_dir)

    if not base_path.exists():
        msg = f"Missing extracted base vectors: {base_path}"
        raise FileNotFoundError(msg)
    return base_path


def _prepare_texmex_records(
    config: DatasetConfig,
    dataset: str,
    cache_path: Path,
) -> tuple[PreparedDatasetManifest, list[PreparedRecord]]:
    """Prepare deterministic records from TexMex base vectors."""

    base_vectors_path = _ensure_texmex_dataset(config, dataset, cache_path / "texmex")
    vectors = _read_fvecs(base_vectors_path)
    total_rows = int(vectors.shape[0])
    if total_rows == 0:
        msg = f"No vectors found in {base_vectors_path}"
        raise ValueError(msg)

    sample_size = min(config.sample_size, total_rows)
    selected_indices = deterministic_sample_ids(
        [str(index) for index in range(total_rows)],
        sample_size=sample_size,
        seed=config.seed,
    )

    prepared_records: list[PreparedRecord] = []
    for selected_index in selected_indices:
        index = int(selected_index)
        record_id = f"{dataset}-{index}"
        metadata = _metadata_fields(record_id, dataset.upper(), dataset.upper(), config.seed)
        prepared_records.append(
            PreparedRecord(
                id=record_id,
                text=f"{dataset} vector {index}",
                embedding=[float(value) for value in vectors[index].tolist()],
                source_dataset=dataset,
                document_title=str(metadata["document_title"]),
                document_length_bucket=str(metadata["document_length_bucket"]),
                categoria=str(metadata["categoria"]),
                tag=str(metadata["tag"]),
                autor=str(metadata["autor"]),
                data_criacao=int(metadata["data_criacao"]),
            )
        )

    manifest = PreparedDatasetManifest(
        dataset_name=dataset,
        dataset_variant="texmex",
        sample_size=len(prepared_records),
        seed=config.seed,
        chunking=None,
    )
    return manifest, prepared_records


def prepare_dataset_from_config(
    config: DatasetConfig,
    cache_dir: str | Path,
    *,
    force_download: bool = False,
) -> tuple[PreparedDatasetManifest, list[PreparedRecord]]:
    """Prepare the configured dataset, using local cache when available."""

    cache_path = Path(cache_dir)
    if not force_download and (cache_path / "manifest.json").exists():
        return load_prepared_dataset(cache_path)

    huggingface_token = os.getenv(config.huggingface_token_env_var)

    if config.primary == "cohere_wikipedia":
        try:
            use_streaming = config.sample_size <= DATASET_FAST_SAMPLE_THRESHOLD
            records = load_huggingface_dataset_records(
                config.cohere_dataset_name,
                config.cohere_dataset_config,
                config.cohere_split,
                streaming=use_streaming,
                max_records=config.sample_size if use_streaming else None,
                token=huggingface_token,
            )
            prepared_records = prepare_cohere_wikipedia_records(
                records,
                config.sample_size,
                config.seed,
            )
            manifest = PreparedDatasetManifest(
                dataset_name=config.cohere_dataset_name,
                dataset_variant=config.cohere_dataset_config,
                sample_size=len(prepared_records),
                seed=config.seed,
                chunking=None,
            )
        except Exception:
            if not config.fallback_enabled:
                raise
            records = load_huggingface_dataset_records(
                config.simple_wikipedia_name,
                config.simple_wikipedia_config,
                "train",
                streaming=True,
                max_records=config.sample_size,
                trust_remote_code=True,
                token=huggingface_token,
            )
            prepared_records = prepare_simple_wikipedia_records(
                records,
                config,
                sentence_transformer_embedder(config.local_embedding_model),
            )
            manifest = PreparedDatasetManifest(
                dataset_name=config.simple_wikipedia_name,
                dataset_variant=config.simple_wikipedia_config,
                sample_size=config.sample_size,
                seed=config.seed,
                chunking=chunker_to_metadata(
                    config.local_embedding_model,
                    config.chunk_size,
                    config.chunk_overlap,
                ),
            )
    elif config.primary == "fast_wikipedia":
        records = load_huggingface_dataset_records(
            config.fast_wikipedia_dataset_name,
            None,
            config.fast_wikipedia_split,
            streaming=True,
            max_records=config.sample_size,
            token=huggingface_token,
        )
        prepared_records = _prepare_fast_wikipedia_records(
            records,
            config.sample_size,
            config.seed,
        )
        manifest = PreparedDatasetManifest(
            dataset_name=config.fast_wikipedia_dataset_name,
            dataset_variant="fast_wikipedia",
            sample_size=len(prepared_records),
            seed=config.seed,
            chunking=None,
        )
    elif config.primary == "simple_wikipedia":
        records = load_huggingface_dataset_records(
            config.simple_wikipedia_name,
            config.simple_wikipedia_config,
            "train",
            streaming=True,
            max_records=config.sample_size,
            trust_remote_code=True,
            token=huggingface_token,
        )
        prepared_records = prepare_simple_wikipedia_records(
            records,
            config,
            sentence_transformer_embedder(config.local_embedding_model),
        )
        manifest = PreparedDatasetManifest(
            dataset_name=config.simple_wikipedia_name,
            dataset_variant=config.simple_wikipedia_config,
            sample_size=config.sample_size,
            seed=config.seed,
            chunking=chunker_to_metadata(
                config.local_embedding_model,
                config.chunk_size,
                config.chunk_overlap,
            ),
        )
    else:
        manifest, prepared_records = _prepare_texmex_records(config, config.primary, cache_path)

    persist_prepared_dataset(cache_path, prepared_records, manifest)
    return manifest, prepared_records


def download_dataset_from_config(
    config: DatasetConfig,
    cache_dir: str | Path,
) -> tuple[PreparedDatasetManifest, list[PreparedRecord]]:
    """Force dataset preparation and local caching for later benchmark runs."""

    return prepare_dataset_from_config(config, cache_dir, force_download=True)
