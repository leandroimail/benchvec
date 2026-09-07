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


def _ivfpq_records(row_count: int, dimension: int = 384) -> list[EngineRecord]:
    """Deterministic records large enough for IVF_PQ training (>= 256 rows)."""

    scale = 1.0 / dimension
    return [
        EngineRecord(
            id=f"row-{index}",
            embedding=[((index * 7 + component * 13) % 97) * scale for component in range(dimension)],
            document=f"doc-{index}",
            metadata={"tag": f"tag-{index % 3}"},
        )
        for index in range(row_count)
    ]


@pytest.mark.slow
def test_lancedb_ivfpq_index_is_built_and_used(tmp_path: Path) -> None:
    engine = LanceDbEngine(
        tmp_path / "lancedb",
        distance_metric="l2",
        index_type="ivfpq",
        num_partitions=2,
        num_sub_vectors=48,
        nprobes=1,
        refine_factor=1,
    )
    engine.reset()
    records = _ivfpq_records(256)
    engine.insert(records)
    engine.build_index()

    # Assert the index physically exists (recall cannot prove this: a scan
    # is also exact, so the check must inspect index metadata).
    assert engine._effective_index_type == "ivfpq"
    assert len(list(engine._table.list_indices())) == 1

    hits = engine.search(records[0].embedding, k=5, metric="l2")
    assert len(hits) == 5
    assert hits[0].id == "row-0"
    assert engine.last_search_params["index_type"] == "ivfpq"
    assert engine.last_search_params["nprobes"] == 1
    assert engine.describe_runtime().index_type == "ivfpq"


@pytest.mark.slow
def test_lancedb_ivfpq_index_survives_reopen(tmp_path: Path) -> None:
    storage_path = tmp_path / "lancedb"
    engine = LanceDbEngine(
        storage_path,
        distance_metric="l2",
        index_type="ivfpq",
        num_partitions=2,
        num_sub_vectors=48,
    )
    engine.reset()
    records = _ivfpq_records(256)
    engine.insert(records)
    engine.build_index()
    engine.persist()

    reopened = LanceDbEngine(storage_path, distance_metric="l2", index_type="ivfpq")
    reopened.reopen()

    assert reopened._effective_index_type == "ivfpq"
    hits = reopened.search(records[0].embedding, k=3, metric="l2")
    assert hits[0].id == "row-0"


@pytest.mark.unit
def test_lancedb_ivfpq_skipped_below_training_floor(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    engine = LanceDbEngine(
        tmp_path / "lancedb",
        distance_metric="l2",
        index_type="ivfpq",
        num_partitions=2,
        num_sub_vectors=1,
    )
    engine.reset()
    engine.insert(_records())  # 2 rows << 256-row PQ floor
    with caplog.at_level("WARNING"):
        engine.build_index()

    assert engine._effective_index_type == "scan"
    assert "below training floor" in caplog.text
    hits = engine.search([0.1, 0.1], k=2, metric="l2")
    assert hits[0].id == "a"
    assert engine.last_search_params == {"index_type": "scan"}


@pytest.mark.unit
def test_lancedb_ivfpq_rejects_indivisible_sub_vectors(tmp_path: Path) -> None:
    engine = LanceDbEngine(
        tmp_path / "lancedb",
        distance_metric="l2",
        index_type="ivfpq",
        num_partitions=1,
        num_sub_vectors=5,
    )
    engine.reset()
    engine.insert(_ivfpq_records(300))

    with pytest.raises(ValueError, match="must divide vector dimension"):
        engine.build_index()


@pytest.mark.unit
def test_lancedb_none_index_type_reports_scan(tmp_path: Path) -> None:
    engine = LanceDbEngine(tmp_path / "lancedb", distance_metric="l2", index_type="none")
    engine.reset()
    engine.insert(_ivfpq_records(300))
    engine.build_index()

    assert engine._effective_index_type == "scan"
    assert engine.describe_runtime().index_type == "scan"


@pytest.mark.unit
def test_lancedb_delete_and_upsert(tmp_path: Path) -> None:
    engine = LanceDbEngine(tmp_path / "lancedb", distance_metric="l2")
    engine.reset()
    records = _records()
    engine.insert(records)

    deleted_count = engine.delete(["a", "missing"])

    assert deleted_count == 2
    assert engine.search([0.1, 0.1], k=2, metric="l2")[0].id == "b"

    engine.upsert([records[0]])
    hits = engine.search([0.1, 0.1], k=2, metric="l2")
    assert hits[0].id == "a"


@pytest.mark.unit
def test_lancedb_delete_handles_quote_bearing_ids(tmp_path: Path) -> None:
    engine = LanceDbEngine(tmp_path / "lancedb", distance_metric="l2")
    engine.reset()
    records = [
        EngineRecord(id="it's \"quoted\"", embedding=[1.0, 1.0], document="q", metadata={}),
        EngineRecord(id="plain", embedding=[0.0, 0.0], document="p", metadata={}),
    ]
    engine.insert(records)

    deleted_count = engine.delete(["it's \"quoted\""])

    assert deleted_count == 1
    hits = engine.search([0.0, 0.0], k=2, metric="l2")
    assert [hit.id for hit in hits] == ["plain"]
