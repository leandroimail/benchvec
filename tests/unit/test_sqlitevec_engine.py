"""Unit tests for the sqlite-vec engine adapter."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from benchvec.engines.base import EngineRecord, FilterClause, MetadataFilter
from benchvec.engines.sqlite_vec_engine import SqliteVecEngine, sqlite_vec_available

# Skip the whole module cleanly when the sqlite-vec extension cannot load
# (e.g. SQLite builds without extension-loading support).
pytestmark = pytest.mark.skipif(
    not sqlite_vec_available(),
    reason="sqlite-vec extension cannot be loaded on this host",
)


def _records() -> list[EngineRecord]:
    """Two 2-D records with L2 distances sqrt(0.02) and sqrt(1.62) to [0.1, 0.1]."""

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
def test_sqlitevec_engine_l2_search_and_filter(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(_records())

    # Act
    hits = engine.search([0.1, 0.1], k=2, metric="l2")

    # Assert: exact ordering and exact native vec0 euclidean distances.
    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(math.sqrt(0.02))
    assert hits[1].score == pytest.approx(math.sqrt(1.62))
    assert hits[0].document == "alpha"
    assert hits[0].metadata == {"tag": "engenharia_dados", "year": 2025}
    assert engine.last_search_params == {"index_type": "bruteforce-sqlite"}

    # Act: equality filter via the application-layer (JSON metadata) path.
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

    # Assert
    assert [hit.id for hit in filtered_hits] == ["a"]
    assert filtered_hits[0].score == pytest.approx(math.sqrt(0.02))
    assert engine.last_search_params == {
        "index_type": "bruteforce-sqlite",
        "filter_mode": "application",
    }
    runtime = engine.describe_runtime()
    assert runtime.engine == "sqlite_vec"
    assert runtime.index_type == "bruteforce-sqlite"
    assert runtime.native_filtering is False
    assert runtime.distance_metric == "l2"


@pytest.mark.unit
def test_sqlitevec_engine_persist_reopen_preserves_data(tmp_path: Path) -> None:
    # Arrange
    storage_path = tmp_path / "sqlitevec"
    engine = SqliteVecEngine(storage_path, distance_metric="l2")
    engine.reset()
    engine.insert(_records())
    engine.persist()

    # Act
    reopened = SqliteVecEngine(storage_path, distance_metric="l2")
    reopened.reopen()
    hits = reopened.search([0.1, 0.1], k=1, metric="l2")

    # Assert
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].metadata == {"tag": "engenharia_dados", "year": 2025}
    assert reopened.disk_size() > 0
    assert (storage_path / "benchvec.sqlite").exists()

    # Assert: reopen restored the dimension, so post-reopen inserts of the
    # same dimension are accepted.
    reopened.insert([EngineRecord(id="c", embedding=[2.0, 2.0], document="gamma")])
    assert [hit.id for hit in reopened.search([2.0, 2.0], k=1, metric="l2")] == ["c"]


@pytest.mark.unit
def test_sqlitevec_engine_cosine_metric_exact_distances(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="cosine")
    engine.reset()
    engine.insert(
        [
            EngineRecord(id="x", embedding=[1.0, 0.0]),
            EngineRecord(id="y", embedding=[0.0, 1.0]),
        ]
    )

    # Act
    hits = engine.search([1.0, 0.0], k=2, metric="cosine")

    # Assert: native vec0 cosine distance = 1 - cosine_similarity, exact.
    assert [hit.id for hit in hits] == ["x", "y"]
    assert hits[0].score == 0.0
    assert hits[1].score == 1.0
    assert engine.describe_runtime().distance_metric == "cosine"


@pytest.mark.unit
def test_sqlitevec_engine_filtered_search_prefilters_closer_non_matching(
    tmp_path: Path,
) -> None:
    # Arrange: the non-matching row is the globally closest one.
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(
        [
            EngineRecord(id="near", embedding=[0.5, 0.5], metadata={"tag": "b"}),
            EngineRecord(id="match", embedding=[3.0, 3.0], metadata={"tag": "a"}),
        ]
    )

    # Act
    hits = engine.search_filtered(
        [0.0, 0.0],
        MetadataFilter(clauses=(FilterClause(field="tag", operator="eq", value="a"),)),
        k=2,
        metric="l2",
    )

    # Assert: only the matching row is returned despite being farther away.
    assert [hit.id for hit in hits] == ["match"]
    assert hits[0].score == pytest.approx(math.sqrt(18.0))


@pytest.mark.unit
def test_sqlitevec_engine_numeric_range_filter(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(_records())

    # Act
    hits = engine.search_filtered(
        [0.1, 0.1],
        MetadataFilter(clauses=(FilterClause(field="year", operator="gte", value=2024),)),
        k=2,
        metric="l2",
    )

    # Assert
    assert [hit.id for hit in hits] == ["a"]


@pytest.mark.unit
def test_sqlitevec_engine_delete_removes_exactly_requested_ids(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(_records())

    # Act
    removed = engine.delete(["a"])
    hits_after = engine.search([0.1, 0.1], k=5, metric="l2")
    removed_again = engine.delete(["a"])
    removed_missing = engine.delete(["missing"])

    # Assert: count is exact and only the requested ids disappear.
    assert removed == 1
    assert [hit.id for hit in hits_after] == ["b"]
    assert removed_again == 0
    assert removed_missing == 0


@pytest.mark.unit
def test_sqlitevec_engine_upsert_overwrites_existing_id(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(_records())
    replacement = EngineRecord(
        id="a",
        embedding=[0.95, 0.95],
        document="alpha-2",
        metadata={"tag": "ciencia", "year": 2024},
    )

    # Act
    engine.upsert([replacement])
    hits = engine.search([0.95, 0.95], k=5, metric="l2")

    # Assert: "a" moved nearest (vector replaced), no duplicate rows,
    # and document/metadata side data was overwritten.
    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].document == "alpha-2"
    assert hits[0].metadata == {"tag": "ciencia", "year": 2024}
    assert len({hit.id for hit in hits}) == 2


@pytest.mark.unit
def test_sqlitevec_engine_upsert_creates_schema_on_fresh_engine(tmp_path: Path) -> None:
    # Arrange: never-inserted engine (no tables yet).
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()

    # Act
    engine.upsert(
        [
            EngineRecord(
                id="a",
                embedding=[0.0, 0.0],
                document="doc",
                metadata={"tag": "t"},
            )
        ]
    )
    hits = engine.search([0.0, 0.0], k=1, metric="l2")

    # Assert
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].document == "doc"


@pytest.mark.unit
def test_sqlitevec_engine_dimension_mismatch_raises(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(_records())

    # Act / Assert: native vec0 errors propagate for both inserts and queries.
    with pytest.raises(sqlite3.OperationalError):
        engine.insert([EngineRecord(id="bad", embedding=[1.0, 2.0, 3.0])])
    with pytest.raises(sqlite3.OperationalError):
        engine.search([1.0, 2.0, 3.0], k=1, metric="l2")


@pytest.mark.unit
def test_sqlitevec_engine_empty_search_returns_empty_list(tmp_path: Path) -> None:
    # Arrange: engine without any insert (tables not created yet).
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()

    # Act
    hits = engine.search([0.1, 0.1], k=3, metric="l2")
    filtered_hits = engine.search_filtered(
        [0.1, 0.1],
        MetadataFilter(clauses=(FilterClause(field="tag", operator="eq", value="a"),)),
        k=3,
        metric="l2",
    )

    # Assert
    assert hits == []
    assert filtered_hits == []

    # Arrange: schema created, then fully emptied via delete.
    engine.insert(_records())
    engine.delete(["a", "b"])

    # Act
    hits_after_delete_all = engine.search([0.1, 0.1], k=3, metric="l2")

    # Assert
    assert hits_after_delete_all == []


@pytest.mark.unit
def test_sqlitevec_engine_reset_clears_all_data(tmp_path: Path) -> None:
    # Arrange
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(_records())

    # Act
    engine.reset()
    hits = engine.search([0.1, 0.1], k=2, metric="l2")

    # Assert
    assert hits == []

    # Assert: the schema is recreated lazily, so re-ingestion works after reset.
    engine.insert(_records())
    assert len(engine.search([0.1, 0.1], k=2, metric="l2")) == 2


@pytest.mark.unit
def test_sqlitevec_engine_sparse_metadata_and_optional_document(tmp_path: Path) -> None:
    # Arrange: one record without a document and with metadata missing "year".
    engine = SqliteVecEngine(tmp_path / "sqlitevec", distance_metric="l2")
    engine.reset()
    engine.insert(
        [
            EngineRecord(
                id="full",
                embedding=[0.0, 0.0],
                document="doc",
                metadata={"tag": "a", "year": 2025},
            ),
            EngineRecord(id="sparse", embedding=[1.0, 1.0], metadata={"tag": "a"}),
        ]
    )

    # Act
    hits = engine.search([0.0, 0.0], k=2, metric="l2")
    filtered_hits = engine.search_filtered(
        [0.0, 0.0],
        MetadataFilter(clauses=(FilterClause(field="year", operator="gte", value=1),)),
        k=2,
        metric="l2",
    )

    # Assert: None document and sparse metadata round-trip faithfully.
    assert [hit.id for hit in hits] == ["full", "sparse"]
    assert hits[1].document is None
    assert hits[1].metadata == {"tag": "a"}
    # A filter on a missing field excludes the record entirely.
    assert [hit.id for hit in filtered_hits] == ["full"]
