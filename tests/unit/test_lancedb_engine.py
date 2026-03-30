"""Unit tests for the LanceDB engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.engines.base import EngineRecord, FilterClause, MetadataFilter
from benchvec.engines.lancedb_engine import LanceDbEngine


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
def test_lancedb_engine_search_and_filter(tmp_path: Path) -> None:
    engine = LanceDbEngine(tmp_path / "lancedb", distance_metric="l2")
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
def test_lancedb_engine_reopen_preserves_data(tmp_path: Path) -> None:
    storage_path = tmp_path / "lancedb"
    engine = LanceDbEngine(storage_path, distance_metric="l2")
    engine.reset()
    engine.insert(_records())
    engine.persist()

    reopened = LanceDbEngine(storage_path, distance_metric="l2")
    reopened.reopen()
    hits = reopened.search([0.1, 0.1], k=1, metric="l2")

    assert hits[0].id == "a"
    assert reopened.disk_size() > 0
