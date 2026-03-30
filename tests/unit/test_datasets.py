"""Unit tests for deterministic dataset preparation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.chunker import ChunkedRecord
from benchvec.config import DatasetConfig
from benchvec.datasets import (
    PreparedDatasetManifest,
    canonical_text,
    infer_author,
    infer_category,
    infer_date,
    infer_tag,
    length_bucket,
    load_prepared_dataset,
    persist_prepared_dataset,
    prepare_cohere_wikipedia_records,
    prepare_simple_wikipedia_records,
)


class FakeChunker:
    def __call__(self, text: str) -> list[ChunkedRecord]:
        parts = text.split("\n\n", maxsplit=1)
        if len(parts) == 1:
            return [
                ChunkedRecord(
                    chunk_id="doc_c0",
                    parent_document_id="doc",
                    text=text,
                    token_count=3,
                )
            ]
        return [
            ChunkedRecord(
                chunk_id="doc_c0",
                parent_document_id="doc",
                text=parts[0],
                token_count=2,
            ),
            ChunkedRecord(
                chunk_id="doc_c1",
                parent_document_id="doc",
                text=parts[1],
                token_count=3,
            ),
        ]


def fake_embedder(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


@pytest.mark.unit
def test_metadata_derivation_is_deterministic() -> None:
    record_id = "doc-1"
    title = "Database Systems"
    text = "A database pipeline with sql and analytics"

    assert (
        canonical_text(title, text)
        == "Database Systems\n\nA database pipeline with sql and analytics"
    )
    assert length_bucket("word " * 140) == "medium"
    assert infer_category(record_id, 42, title, text) == "data_engineering"
    assert infer_tag(record_id, 42, "data_engineering") in {
        "engenharia_dados",
        "etl",
        "sql",
        "analytics",
    }
    assert infer_author(record_id, 42) in {
        "engenharia",
        "pesquisa",
        "produto",
        "operacoes",
        "plataforma",
        "arquitetura",
        "dados",
        "ia",
    }
    assert isinstance(infer_date(record_id, 42), int)
    assert infer_date(record_id, 42) > 0


@pytest.mark.unit
def test_prepare_cohere_wikipedia_records_keeps_sample_deterministic() -> None:
    records = [
        {"id": "c", "title": "Gamma", "text": "third", "embedding": [0.3, 0.4]},
        {"id": "a", "title": "Alpha", "text": "first", "embedding": [0.1, 0.2]},
        {"id": "b", "title": "Beta", "text": "second", "embedding": [0.2, 0.3]},
    ]

    prepared = prepare_cohere_wikipedia_records(records, sample_size=2, seed=7)

    assert len(prepared) == 2
    assert all(record.source_dataset == "cohere_wikipedia" for record in prepared)
    assert all(len(record.embedding) == 2 for record in prepared)


@pytest.mark.unit
def test_prepare_simple_wikipedia_records_chunks_and_embeds() -> None:
    config = DatasetConfig(
        primary="simple_wikipedia",
        sample_size=1,
        seed=5,
        min_text_characters=10,
    )
    records = [{"id": "doc", "title": "Title", "text": "Some sufficiently large body of text"}]

    prepared = prepare_simple_wikipedia_records(
        records,
        config,
        fake_embedder,
        chunker=FakeChunker(),
    )

    assert len(prepared) == 2
    assert prepared[0].parent_document_id == "doc"
    assert prepared[0].source_dataset == "simple_wikipedia"
    assert prepared[0].embedding


@pytest.mark.unit
def test_persist_and_load_prepared_dataset_round_trip(tmp_path: Path) -> None:
    records = prepare_cohere_wikipedia_records(
        [{"id": "doc-1", "title": "Alpha", "text": "Example", "embedding": [0.1, 0.2]}],
        sample_size=1,
        seed=1,
    )
    manifest = PreparedDatasetManifest(
        dataset_name="Cohere/wikipedia-22-12-simple-embeddings",
        dataset_variant="en",
        sample_size=1,
        seed=1,
    )

    persist_prepared_dataset(tmp_path, records, manifest)
    loaded_manifest, loaded_records = load_prepared_dataset(tmp_path)

    assert loaded_manifest.dataset_name == manifest.dataset_name
    assert len(loaded_records) == 1
    assert loaded_records[0].id == "doc-1"
