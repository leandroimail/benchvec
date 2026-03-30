"""Unit tests for the FAISS engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.engines.base import EngineRecord, FilterClause, MetadataFilter
from benchvec.engines.faiss_engine import FaissEngine


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
def test_faiss_engine_search_and_filter(tmp_path: Path) -> None:
    engine = FaissEngine(tmp_path / "faiss", distance_metric="l2", index_type="flat")
    engine.connect()
    engine.insert(_records())
    engine.build_index()

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
    assert engine.describe_runtime().engine == "faiss"


@pytest.mark.unit
def test_faiss_engine_persist_and_reopen(tmp_path: Path) -> None:
    storage_path = tmp_path / "faiss"
    engine = FaissEngine(storage_path, distance_metric="l2", index_type="flat")
    engine.connect()
    engine.insert(_records())
    engine.persist()

    reopened = FaissEngine(storage_path, distance_metric="l2", index_type="flat")
    reopened.reopen()
    hits = reopened.search([0.1, 0.1], k=1, metric="l2")

    assert hits[0].id == "a"
    assert reopened.disk_size() > 0
