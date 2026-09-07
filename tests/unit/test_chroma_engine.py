"""Unit tests for the ChromaDB engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.engines.base import EngineRecord, FilterClause, MetadataFilter
from benchvec.engines.chroma_engine import ChromaEngine


def _records() -> list[EngineRecord]:
    return [
        EngineRecord(
            id="a",
            embedding=[0.0, 0.0],
            document="alpha",
            metadata={"tag": "engenharia_dados", "year": 2025},
        ),
        EngineRecord(
            id="b",
            embedding=[1.0, 1.0],
            document="beta",
            metadata={"tag": "ciencia", "year": 2023},
        ),
    ]


@pytest.mark.unit
def test_chroma_engine_search_and_filter(tmp_path: Path) -> None:
    engine = ChromaEngine(tmp_path / "chroma", distance_metric="l2")
    engine.reset()
    engine.insert(_records())

    hits = engine.search([0.1, 0.1], k=2, metric="l2")
    filtered_hits = engine.search_filtered(
        [0.1, 0.1],
        MetadataFilter(
            clauses=(
                FilterClause(field="tag", operator="eq", value="engenharia_dados"),
            )
        ),
        k=2,
        metric="l2",
    )

    assert hits[0].id == "a"
    assert filtered_hits[0].id == "a"
    assert engine.describe_runtime().native_filtering is True


@pytest.mark.unit
def test_chroma_engine_reopen_preserves_data(tmp_path: Path) -> None:
    storage_path = tmp_path / "chroma"
    engine = ChromaEngine(storage_path, distance_metric="l2")
    engine.reset()
    engine.insert(_records())
    engine.persist()

    reopened = ChromaEngine(storage_path, distance_metric="l2")
    reopened.reopen()
    hits = reopened.search([0.1, 0.1], k=1, metric="l2")

    assert hits[0].id == "a"
    assert reopened.disk_size() > 0


@pytest.mark.unit
def test_chroma_ef_search_recorded_in_search_params(tmp_path: Path) -> None:
    engine = ChromaEngine(tmp_path / "chroma", distance_metric="l2", search_ef=32)
    engine.reset()
    engine.insert(_records())

    hits = engine.search([0.1, 0.1], k=2, metric="l2")

    assert len(hits) == 2
    assert engine.last_search_params == {"ef_search": 32}
    assert engine.ef_creation_mode in {"configuration", "legacy-metadata"}


@pytest.mark.unit
def test_chroma_rebuild_with_ef_updates_params(tmp_path: Path) -> None:
    engine = ChromaEngine(tmp_path / "chroma", distance_metric="l2", search_ef=32)
    engine.reset()
    engine.insert(_records())

    engine.rebuild_with_ef(96)

    assert engine.search_ef == 96
    assert engine.ef_creation_mode in {"configuration", "legacy-metadata"}
    # Data was dropped with the collection; engine must be re-ingested.
    assert engine.search([0.1, 0.1], k=2, metric="l2") == []


@pytest.mark.unit
def test_chroma_fallback_to_legacy_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = ChromaEngine(tmp_path / "chroma", distance_metric="l2", search_ef=32)
    engine.reset()
    assert engine.ef_creation_mode == "configuration"

    legacy_sentinel = object()

    def reject_configuration(*args: object, **kwargs: object) -> object:
        if "configuration" in kwargs:
            raise TypeError("configuration not supported")
        assert "metadata" in kwargs and "hnsw:search_ef" in kwargs["metadata"]
        return legacy_sentinel

    class FallbackClient:
        def get_or_create_collection(self, *args: object, **kwargs: object) -> object:
            return reject_configuration(*args, **kwargs)

    import chromadb

    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda *args, **kwargs: FallbackClient(),
    )

    engine.connect()

    assert engine.ef_creation_mode == "legacy-metadata"


@pytest.mark.unit
def test_chroma_delete_and_upsert(tmp_path: Path) -> None:
    engine = ChromaEngine(tmp_path / "chroma", distance_metric="l2")
    engine.reset()
    records = _records()
    engine.insert(records)

    deleted_count = engine.delete(["a", "missing"])

    assert deleted_count == 2
    assert engine.search([0.1, 0.1], k=2, metric="l2")[0].id == "b"

    engine.upsert([records[0]])
    hits = engine.search([0.1, 0.1], k=2, metric="l2")
    assert hits[0].id == "a"
