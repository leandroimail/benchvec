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


def _ivf_records() -> list[EngineRecord]:
    """256 clustered records: enough rows for every nlist used in tests."""

    return [
        EngineRecord(
            id=f"row-{index}",
            embedding=[float(index % 16), float(index // 16)],
            document=f"doc-{index}",
            metadata={"tag": f"tag-{index % 2}"},
        )
        for index in range(256)
    ]


@pytest.mark.unit
def test_faiss_ivfflat_applies_configured_nprobe(tmp_path: Path) -> None:
    engine = FaissEngine(
        tmp_path / "faiss",
        distance_metric="l2",
        index_type="ivfflat",
        nlist=4,
        nprobe=2,
    )
    engine.connect()
    engine.insert(_ivf_records())
    engine.build_index()

    hits = engine.search([0.0, 0.0], k=3, metric="l2")

    assert engine._index.nprobe == 2
    assert engine.last_search_params == {"nprobe": 2}
    assert len(hits) == 3


@pytest.mark.unit
def test_faiss_ivfflat_set_query_params_updates_nprobe_live(tmp_path: Path) -> None:
    engine = FaissEngine(
        tmp_path / "faiss",
        distance_metric="l2",
        index_type="ivfflat",
        nlist=4,
        nprobe=1,
    )
    engine.connect()
    engine.insert(_ivf_records())
    engine.build_index()

    engine.set_query_params({"nprobe": 4})
    engine.search([0.0, 0.0], k=3, metric="l2")

    assert engine._index.nprobe == 4
    assert engine.last_search_params == {"nprobe": 4}


@pytest.mark.unit
def test_faiss_ivfflat_zero_nprobe_keeps_library_default(tmp_path: Path) -> None:
    engine = FaissEngine(
        tmp_path / "faiss",
        distance_metric="l2",
        index_type="ivfflat",
        nlist=4,
        nprobe=0,
    )
    engine.connect()
    engine.insert(_ivf_records())
    engine.build_index()
    library_default = engine._index.nprobe

    engine.search([0.0, 0.0], k=3, metric="l2")

    assert engine._index.nprobe == library_default == 1


@pytest.mark.unit
def test_faiss_flat_ignores_nprobe_param(tmp_path: Path) -> None:
    engine = FaissEngine(tmp_path / "faiss", distance_metric="l2", index_type="flat")
    engine.connect()
    engine.insert(_records())
    engine.build_index()

    engine.set_query_params({"nprobe": 8})
    engine.search([0.1, 0.1], k=2, metric="l2")

    assert engine.last_search_params == {}


@pytest.mark.unit
def test_faiss_delete_tombstones_and_upsert_restores(tmp_path: Path) -> None:
    engine = FaissEngine(tmp_path / "faiss", distance_metric="l2", index_type="flat")
    engine.connect()
    records = _ivf_records()
    engine.insert(records)
    engine.build_index()

    deleted_count = engine.delete(["row-0", "row-1", "missing-id"])

    assert deleted_count == 2
    hits = engine.search([float(0), float(0)], k=5, metric="l2")
    hit_ids = {hit.id for hit in hits}
    assert "row-0" not in hit_ids
    assert "row-1" not in hit_ids

    engine.upsert([records[0]])
    hits_after = engine.search([float(0), float(0)], k=5, metric="l2")
    assert "row-0" in {hit.id for hit in hits_after}


@pytest.mark.unit
def test_faiss_tombstones_survive_reopen(tmp_path: Path) -> None:
    storage_path = tmp_path / "faiss"
    engine = FaissEngine(storage_path, distance_metric="l2", index_type="flat")
    engine.connect()
    records = _ivf_records()
    engine.insert(records)
    engine.delete(["row-0"])
    engine.persist()

    reopened = FaissEngine(storage_path, distance_metric="l2", index_type="flat")
    reopened.reopen()
    hits = reopened.search([float(0), float(0)], k=5, metric="l2")

    assert "row-0" not in {hit.id for hit in hits}


@pytest.mark.unit
def test_faiss_flat_upsert_replaces_vector_without_duplicates(tmp_path: Path) -> None:
    engine = FaissEngine(tmp_path / "faiss", distance_metric="l2", index_type="flat")
    engine.connect()
    records = _ivf_records()
    engine.insert(records)
    engine.build_index()

    replaced = EngineRecord(
        id="row-0",
        embedding=[100.0, 100.0],
        document="replaced",
        metadata={"tag": "replaced"},
    )
    engine.upsert([replaced])

    hits_near_old_vector = engine.search([0.0, 0.0], k=len(records), metric="l2")
    hit_ids = [hit.id for hit in hits_near_old_vector]
    # The stale [0, 0] row must not resurface…
    assert hits_near_old_vector[0].id != "row-0"
    # …and the id must appear at most once (no physical-row duplicates).
    assert hit_ids.count("row-0") <= 1
    # The new vector is the only remaining row near [100, 100].
    hits_near_new_vector = engine.search([100.0, 100.0], k=1, metric="l2")
    assert hits_near_new_vector[0].id == "row-0"


@pytest.mark.unit
def test_faiss_ivfflat_post_build_upsert_is_searchable(tmp_path: Path) -> None:
    engine = FaissEngine(
        tmp_path / "faiss",
        distance_metric="l2",
        index_type="ivfflat",
        nlist=4,
    )
    engine.connect()
    records = _ivf_records()
    engine.insert(records)
    engine.build_index()

    replaced = EngineRecord(
        id="row-0",
        embedding=[100.0, 100.0],
        document="replaced",
        metadata={"tag": "replaced"},
    )
    engine.upsert([replaced])

    hits = engine.search([100.0, 100.0], k=3, metric="l2")
    assert hits[0].id == "row-0"
    filtered_hits = engine.search_filtered(
        [100.0, 100.0],
        MetadataFilter(clauses=(FilterClause(field="tag", operator="eq", value="replaced"),)),
        k=3,
        metric="l2",
    )
    assert filtered_hits[0].id == "row-0"


@pytest.mark.unit
def test_faiss_delete_does_not_double_count(tmp_path: Path) -> None:
    engine = FaissEngine(tmp_path / "faiss", distance_metric="l2", index_type="flat")
    engine.connect()
    engine.insert(_records())

    first = engine.delete(["a"])
    second = engine.delete(["a"])

    assert first == 1
    assert second == 0
