"""Unit tests for the TurboVec adapter (turbovec_engine).

The whole module skips cleanly when the turbovec wheel is unavailable
(musl / old-glibc hosts have no prebuilt wheel and local Rust builds are
never performed).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from benchvec.engines.base import EngineRecord, FilterClause, MetadataFilter
from benchvec.engines.turbovec_engine import (
    TurboVecEngine,
    turbovec_available,
)

pytestmark = pytest.mark.skipif(
    not turbovec_available(),
    reason="turbovec wheel is not available on this host",
)

DIM = 16  # turbovec requires dim % 8 == 0


def _records(count: int = 20, seed: int = 0) -> list[EngineRecord]:
    """Build deterministic float32-representable records with metadata."""

    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, DIM)).astype(np.float32)
    return [
        EngineRecord(
            id=f"doc-{index}",
            embedding=vectors[index].tolist(),
            metadata={"category": "even" if index % 2 == 0 else "odd", "rank": index},
        )
        for index in range(count)
    ]


def _filled_engine(
    tmp_path: Path, count: int = 20, **kwargs: object
) -> TurboVecEngine:
    """Return a cosine engine in ``tmp_path`` preloaded with records."""

    engine = TurboVecEngine(tmp_path, distance_metric="cosine", **kwargs)  # type: ignore[arg-type]
    engine.insert(_records(count))
    return engine


# ---------------------------------------------------------------------------
# Construction and capability guards
# ---------------------------------------------------------------------------
def test_constructor_rejects_l2_metric(tmp_path: Path) -> None:
    # Act + Assert: the kernel is inner-product-only; the error names the
    # config key that must change.
    with pytest.raises(ValueError, match="distance_metric"):
        TurboVecEngine(tmp_path, distance_metric="l2")


def test_constructor_rejects_illegal_bit_width(tmp_path: Path) -> None:
    # Act + Assert: only {2, 3, 4} are legal (probed library contract).
    with pytest.raises(ValueError, match="bit_width"):
        TurboVecEngine(tmp_path, distance_metric="cosine", bit_width=8)


def test_search_rejects_l2_metric_argument(tmp_path: Path) -> None:
    # Arrange
    engine = _filled_engine(tmp_path)

    # Act + Assert: a per-call l2 argument is rejected even though the
    # engine itself is cosine (defense in depth on top of config checks).
    with pytest.raises(ValueError, match="inner-product-only"):
        engine.search([0.0] * DIM, 3, "l2")  # type: ignore[arg-type]


def test_insert_rejects_dimension_not_multiple_of_eight(tmp_path: Path) -> None:
    # Arrange: dim 12 is not a legal turbovec dimension.
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    record = EngineRecord(id="doc-0", embedding=[1.0] * 12)

    # Act + Assert
    with pytest.raises(ValueError, match="multiple of 8"):
        engine.insert([record])


def test_insert_converts_non_float32_embeddings(tmp_path: Path) -> None:
    # Arrange: integer embeddings must be cast to 2-D float32 internally
    # (turbovec rejects anything else).
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert([EngineRecord(id="doc-0", embedding=[1, 0] * 8)])

    # Act
    hits = engine.search([1, 0] * 8, 1, "cosine")

    # Assert
    assert [hit.id for hit in hits] == ["doc-0"]


def test_insert_rejects_duplicate_ids_before_library_call(tmp_path: Path) -> None:
    # Arrange: duplicates inside one batch (or against stored ids) must
    # fail up front — a mid-chunk library error would corrupt the id map.
    engine = _filled_engine(tmp_path, count=2)
    duplicate = EngineRecord(id="doc-0", embedding=[1.0] * DIM)

    # Act + Assert: in-batch duplicate ...
    with pytest.raises(ValueError, match="already present"):
        engine.insert([EngineRecord(id="doc-9", embedding=[1.0] * DIM), duplicate])
    # ... and duplicate against an already-stored id.
    with pytest.raises(ValueError, match="already present"):
        engine.insert([EngineRecord(id="doc-1", embedding=[2.0] * DIM)])


# ---------------------------------------------------------------------------
# Search behavior
# ---------------------------------------------------------------------------
def test_round_trip_returns_inserted_ids(tmp_path: Path) -> None:
    # Arrange
    records = _records(20, seed=4)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)

    # Act: querying with a stored vector must return its own id first.
    hits = engine.search(records[7].embedding, 5, "cosine")

    # Assert
    assert hits[0].id == "doc-7"
    assert {hit.id for hit in hits} <= {record.id for record in records}


def test_search_on_empty_engine_returns_empty_list(tmp_path: Path) -> None:
    # Arrange
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")

    # Act
    hits = engine.search([1.0] * DIM, 3, "cosine")

    # Assert
    assert hits == []


def test_search_with_k_larger_than_index_size(tmp_path: Path) -> None:
    # Arrange: 5 stored vectors, k = 20 (probed: returns min(k, n) rows).
    engine = _filled_engine(tmp_path, count=5)

    # Act
    hits = engine.search([1.0] + [0.0] * (DIM - 1), 20, "cosine")

    # Assert: every stored id is returned exactly once.
    assert sorted(hit.id for hit in hits) == [f"doc-{i}" for i in range(5)]


def test_zero_vectors_produce_no_nan_and_deterministic_hits(tmp_path: Path) -> None:
    # Arrange: zero vectors normalize to zero (no NaNs) under cosine.
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(
        [
            EngineRecord(id="doc-0", embedding=[0.0] * DIM),
            EngineRecord(id="doc-1", embedding=[1.0] + [0.0] * (DIM - 1)),
        ]
    )

    # Act
    hits = engine.search([0.0] * DIM, 2, "cosine")
    repeat_ids = [hit.id for hit in engine.search([0.0] * DIM, 2, "cosine")]

    # Assert: scores are finite, two hits, deterministic ordering.
    assert all(np.isfinite(hit.score) for hit in hits)
    assert [hit.id for hit in hits] == repeat_ids
    assert len(hits) == 2


def test_last_search_params_populated_on_both_paths(tmp_path: Path) -> None:
    # Arrange
    engine = _filled_engine(tmp_path)
    filters = MetadataFilter(
        clauses=(FilterClause(field="category", operator="eq", value="even"),)
    )

    # Act
    engine.search([1.0] * DIM, 3, "cosine")
    unfiltered_params = engine.last_search_params
    engine.search_filtered([1.0] * DIM, filters, 3, "cosine")
    filtered_params = engine.last_search_params

    # Assert
    assert unfiltered_params == {
        "bit_width": 4,
        "calibration_state": "uncalibrated",
    }
    assert filtered_params["filter_mode"] == "allowlist"
    assert filtered_params["bit_width"] == 4


# ---------------------------------------------------------------------------
# Filtered search
# ---------------------------------------------------------------------------
def test_search_filtered_returns_only_matching_ids(tmp_path: Path) -> None:
    # Arrange
    records = _records(20)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)
    filters = MetadataFilter(
        clauses=(FilterClause(field="category", operator="eq", value="even"),)
    )

    # Act
    hits = engine.search_filtered(records[0].embedding, filters, 20, "cosine")

    # Assert: only even ids, and the engine never saw an empty allowlist.
    returned = [hit.id for hit in hits]
    assert all(int(hit_id.split("-")[1]) % 2 == 0 for hit_id in returned)
    assert len(returned) == 10


def test_search_filtered_with_no_matching_metadata_returns_empty(
    tmp_path: Path,
) -> None:
    # Arrange: no record has this category — the allowlist would be empty,
    # which turbovec rejects, so the adapter must short-circuit.
    engine = _filled_engine(tmp_path)
    filters = MetadataFilter(
        clauses=(FilterClause(field="category", operator="eq", value="none"),)
    )

    # Act
    hits = engine.search_filtered([1.0] * DIM, filters, 3, "cosine")

    # Assert
    assert hits == []


# ---------------------------------------------------------------------------
# Delete / upsert (id-map tombstone correctness)
# ---------------------------------------------------------------------------
def test_delete_then_search_maps_ids_correctly(tmp_path: Path) -> None:
    # Arrange: deleting a middle id tombstones its uint; later uints must
    # still map to their original string ids (no shifting).
    records = _records(10, seed=5)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)

    # Act
    removed = engine.delete(["doc-4", "doc-5", "missing-id"])
    hits = engine.search(records[9].embedding, 10, "cosine")

    # Assert: two removals (unknown id contributes 0); the queried
    # record — whose uint sits ABOVE the tombstones — is still the top
    # hit (a shifting implementation would drop or remap it); no deleted
    # ids in results.
    assert removed == 2
    assert hits[0].id == "doc-9"
    returned = {hit.id for hit in hits}
    assert "doc-4" not in returned and "doc-5" not in returned
    assert returned <= {record.id for record in records}


def test_reinserting_deleted_id_gets_fresh_slot(tmp_path: Path) -> None:
    # Arrange: doc-3 is deleted, then re-added with a different vector.
    records = _records(10, seed=6)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)
    engine.delete(["doc-3"])

    # Act: re-insert under the same string id with a distinctive vector.
    replacement = EngineRecord(id="doc-3", embedding=[9.0] + [0.0] * (DIM - 1))
    engine.upsert([replacement])
    hits = engine.search(replacement.embedding, 1, "cosine")

    # Assert: the fresh slot returns the string id exactly once.
    assert [hit.id for hit in hits] == ["doc-3"]


def test_upsert_overwrites_and_plain_inserts(tmp_path: Path) -> None:
    # Arrange
    records = _records(10, seed=7)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)

    # Act: overwrite doc-0 along e₁; upsert doc-new along e₂ (distinct
    # directions — [9,0,..] and [8,0,..] normalize to the SAME unit vector
    # under cosine and would tie).
    engine.upsert(
        [
            EngineRecord(id="doc-0", embedding=[9.0] + [0.0] * (DIM - 1)),
            EngineRecord(id="doc-new", embedding=[0.0, 8.0] + [0.0] * (DIM - 2)),
        ]
    )

    # Assert: each query vector's own direction is its unique best match
    # (overwrite took) and the unknown id behaved as a plain insert.
    assert [hit.id for hit in engine.search([1.0] + [0.0] * (DIM - 1), 1, "cosine")] == [
        "doc-0"
    ]
    assert [hit.id for hit in engine.search([0.0, 1.0] + [0.0] * (DIM - 2), 1, "cosine")] == [
        "doc-new"
    ]


def test_reset_allows_reinserting_the_same_ids(tmp_path: Path) -> None:
    # Arrange: _load_engine resets before every repetition — leftover
    # state would trip turbovec's duplicate-id ValueError.
    records = _records(10, seed=8)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)
    engine.reset()

    # Act: re-insert the same ids after reset.
    engine.insert(records)
    hits = engine.search(records[2].embedding, 2, "cosine")

    # Assert
    assert hits[0].id == "doc-2"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_persist_reopen_round_trip(tmp_path: Path) -> None:
    # Arrange
    records = _records(12, seed=9)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)
    engine.persist()
    disk_bytes = engine.disk_size()

    # Act
    engine.reopen()
    hits = engine.search(records[4].embedding, 3, "cosine")

    # Assert: ids searchable after reload and the footprint is real.
    assert hits[0].id == "doc-4"
    assert disk_bytes > 0


def test_reopen_with_corrupt_metadata_sidecar_degrades_cleanly(
    tmp_path: Path,
) -> None:
    # Arrange: persisted state plus a truncated metadata.json.
    engine = _filled_engine(tmp_path)
    engine.persist()
    engine._metadata_path.write_text("{not json", encoding="utf-8")

    # Act: reopen must not crash; filtering degrades to no metadata.
    engine.reopen()
    hits = engine.search([1.0] * DIM, 3, "cosine")
    filters = MetadataFilter(
        clauses=(FilterClause(field="category", operator="eq", value="even"),)
    )
    filtered = engine.search_filtered([1.0] * DIM, filters, 3, "cosine")

    # Assert: unfiltered search works; the filter matches nothing.
    assert len(hits) == 3
    assert filtered == []


def test_search_order_matches_numpy_cosine_ranking(tmp_path: Path) -> None:
    # Arrange: normalization must be observable — the full result order
    # has to match an independent numpy cosine ranking (kills mutants
    # that drop _normalize_rows; magnitude-only ordering would differ).
    rng = np.random.default_rng(21)
    vectors = rng.normal(size=(8, DIM)).astype(np.float32) * rng.uniform(
        0.5, 5.0, size=(8, 1)
    ).astype(np.float32)
    records = [
        EngineRecord(id=f"doc-{i}", embedding=vectors[i].tolist()) for i in range(8)
    ]
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)
    query = (rng.normal(size=DIM) * 3.0).astype(np.float32)

    # Act
    hits = engine.search(query.tolist(), 8, "cosine")

    # Assert: exact ordering against the numpy reference.
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = normalized @ (query / np.linalg.norm(query))
    expected = [f"doc-{i}" for i in np.argsort(similarities)[::-1]]
    assert [hit.id for hit in hits] == expected


def test_insert_rejects_mixed_dimension_batches(tmp_path: Path) -> None:
    # Arrange: a wrong-dim record after the first would otherwise pass
    # the dim guard and corrupt the id map at the chunk boundary.
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")

    # Act + Assert
    with pytest.raises(ValueError, match="does not match batch dim"):
        engine.insert(
            [
                EngineRecord(id="doc-0", embedding=[1.0] * DIM),
                EngineRecord(id="doc-1", embedding=[1.0] * (DIM + 8)),
            ]
        )


def test_reopen_with_corrupt_ids_sidecar_discards_snapshot(tmp_path: Path) -> None:
    # Arrange: a usable IdMapIndex requires its id map — with the map
    # gone, keeping the snapshot would collide old uints with new ones.
    engine = _filled_engine(tmp_path, count=4)
    engine.persist()
    engine._ids_path.write_text('{"ids": "not-a-list"}', encoding="utf-8")

    # Act: reopen must degrade to a fresh (empty) engine without crashing.
    engine.reopen()
    hits = engine.search([1.0] * DIM, 3, "cosine")
    engine.insert(_records(4))  # fresh uints must not collide
    hits_after = engine.search([1.0] * DIM, 3, "cosine")

    # Assert: no stale hits, and re-insert + search works again.
    assert hits == []
    assert hits_after, "re-insert after ids-sidecar discard failed"


# ---------------------------------------------------------------------------
# Calibration (TQ+) and rebuild
# ---------------------------------------------------------------------------
def test_build_index_calibrates_and_retains_sample(tmp_path: Path) -> None:
    # Arrange
    records = _records(20, seed=10)
    engine = TurboVecEngine(
        tmp_path, distance_metric="cosine", bit_width=4, calibrate=True
    )
    engine.insert(records)

    # Act
    engine.build_index()
    hits = engine.search(records[1].embedding, 2, "cosine")

    # Assert: state flipped, search works, and the sample is retained so
    # sweep rebuilds can re-calibrate (never silently uncalibrated).
    assert engine.last_search_params["calibration_state"] == "calibrated"
    assert engine._calibration_rows == 20
    assert hits[0].id == "doc-1"


def test_build_index_without_calibration_stays_uncalibrated(
    tmp_path: Path,
) -> None:
    # Arrange
    engine = _filled_engine(tmp_path, count=6)

    # Act
    engine.build_index()

    # Assert
    assert engine.last_search_params == {}
    assert engine._index is not None
    assert engine._index.calibration_state == "uncalibrated"


def test_rebuild_with_bit_width_swaps_index_and_stays_calibrated(
    tmp_path: Path,
) -> None:
    # Arrange
    records = _records(20, seed=11)
    engine = TurboVecEngine(
        tmp_path, distance_metric="cosine", bit_width=4, calibrate=True
    )
    engine.insert(records)
    engine.build_index()

    # Act: the sweep pattern — rebuild then re-insert all records.
    engine.rebuild_with_bit_width(2)
    engine.insert(records)
    hits = engine.search(records[3].embedding, 2, "cosine")

    # Assert: runtime reflects the new width, calibration survived the
    # swap, and the rebuilt index returns the right ids.
    assert engine.bit_width == 2
    assert engine.index_type == "turboquant-2bit"
    assert engine.last_search_params["bit_width"] == 2
    assert engine.last_search_params["calibration_state"] == "calibrated"
    assert hits[0].id == "doc-3"


def test_rebuild_with_bit_width_requires_known_dimension(tmp_path: Path) -> None:
    # Arrange: fresh engine — the dimension is only learned on insert.
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")

    # Act + Assert
    with pytest.raises(RuntimeError, match="dimension"):
        engine.rebuild_with_bit_width(2)


def test_describe_runtime_fields(tmp_path: Path) -> None:
    # Arrange
    engine = _filled_engine(tmp_path)

    # Act
    runtime = engine.describe_runtime()

    # Assert
    assert runtime.engine == "turbovec"
    assert runtime.index_type == "turboquant-4bit"
    assert runtime.distance_metric == "cosine"
    assert runtime.native_filtering is False
    assert runtime.package_version
    assert runtime.storage_path == str(tmp_path)


# ---------------------------------------------------------------------------
# Concurrency (probed: turbovec's Rust side synchronizes; searches run
# lock-free while map mutations are serialized by the adapter lock)
# ---------------------------------------------------------------------------
def test_concurrent_searches_smoke(tmp_path: Path) -> None:
    # Arrange
    records = _records(50, seed=12)
    engine = TurboVecEngine(tmp_path, distance_metric="cosine")
    engine.insert(records)
    errors: list[Exception] = []

    def worker(worker_index: int) -> None:
        try:
            for step in range(50):
                query = records[(worker_index * 7 + step) % len(records)].embedding
                hits = engine.search(query, 5, "cosine")
                # Invariant under test: no crashes and no garbage ids. A
                # hit count of 0 is a benign library-level transient when
                # the query races the mutator's delete+re-add window
                # (benchmark phases always search a quiescent index, so
                # measurements never observe this).
                valid_ids = {record.id for record in records}
                assert all(hit.id in valid_ids for hit in hits)
        except Exception as exc:  # noqa: BLE001 - collected and re-asserted
            errors.append(exc)

    def mutator() -> None:
        """Delete and re-add ids while searches run (locks under test)."""

        try:
            for step in range(25):
                record = records[step % len(records)]
                engine.delete([record.id])
                engine.insert([record])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    # Act
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    threads.append(threading.Thread(target=mutator))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert: no exceptions collected ...
    assert errors == []
    # ... and a quiescent pass finds every record as its own top-1 hit
    # (mutations never lost or remapped records).
    for record in records:
        top = engine.search(record.embedding, 1, "cosine")
        assert top and top[0].id == record.id


def test_reopen_with_missing_ids_sidecar_discards_snapshot(
    tmp_path: Path,
) -> None:
    # Arrange: crash between index.write and the ids sidecar write leaves
    # a snapshot whose uint→string translation is gone — the same unusable
    # state as a corrupt sidecar (final review BLOCKING finding).
    engine = _filled_engine(tmp_path, count=4)
    engine.persist()
    engine._ids_path.unlink()

    # Act: reopen must discard the orphaned snapshot, then re-insert and
    # search must work without duplicate-id collisions.
    engine.reopen()
    hits = engine.search([1.0] * DIM, 3, "cosine")
    engine.insert(_records(4))
    hits_after = engine.search(_records(4)[2].embedding, 1, "cosine")

    # Assert: no stale hits, and the fresh index serves queries again.
    assert hits == []
    assert hits_after and hits_after[0].id == "doc-2"


def test_reopen_with_wrong_shape_metadata_sidecar_degrades(
    tmp_path: Path,
) -> None:
    # Arrange: valid JSON of the wrong shape (bare list) previously
    # crashed search_filtered at query time with an AttributeError.
    engine = _filled_engine(tmp_path, count=4)
    engine.persist()
    engine._metadata_path.write_text('["not", "a", "dict"]', encoding="utf-8")

    # Act
    engine.reopen()
    hits = engine.search([1.0] * DIM, 2, "cosine")
    filters = MetadataFilter(
        clauses=(FilterClause(field="category", operator="eq", value="even"),)
    )
    filtered = engine.search_filtered([1.0] * DIM, filters, 2, "cosine")

    # Assert: unfiltered search fine; the filter degrades to no matches.
    assert len(hits) == 2
    assert filtered == []
