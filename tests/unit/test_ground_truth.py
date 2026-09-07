"""Unit tests for exact ground-truth computation and caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchvec.ground_truth import (
    GroundTruthRow,
    compute_exact_ground_truth,
    load_ground_truth,
    load_ground_truth_as_map,
    persist_ground_truth,
)
from tests.unit.test_runner import build_runner, make_prepared_records


@pytest.mark.unit
def test_compute_exact_ground_truth_returns_expected_neighbors() -> None:
    rows = compute_exact_ground_truth(
        base_ids=["a", "b", "c"],
        base_vectors=[[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]],
        query_ids=["q1"],
        query_vectors=[[0.2, 0.2]],
        k=2,
    )

    assert [row.neighbor_id for row in rows] == ["a", "b"]
    assert [row.rank for row in rows] == [1, 2]


@pytest.mark.unit
def test_persist_and_load_ground_truth_round_trip(tmp_path: Path) -> None:
    rows = compute_exact_ground_truth(
        base_ids=["a", "b"],
        base_vectors=[[0.0, 0.0], [1.0, 1.0]],
        query_ids=["q1"],
        query_vectors=[[0.1, 0.1]],
        k=2,
    )

    persist_ground_truth(tmp_path, rows, dataset_name="cohere", k=2)
    loaded_rows = load_ground_truth(tmp_path)
    loaded_map = load_ground_truth_as_map(tmp_path)

    assert len(loaded_rows) == 2
    assert loaded_map["q1"] == ["a", "b"]


@pytest.mark.unit
def test_cosine_ranks_by_direction_not_magnitude() -> None:
    # Arrange: q=[3,4] — l2 prefers a=[1,0] (dist 20 < 45) while cosine
    # prefers b=[0,10] (cos 0.8 > cos 0.6). The rankings genuinely differ.
    common_kwargs: dict[str, object] = dict(
        base_ids=["a", "b"],
        base_vectors=[[1.0, 0.0], [0.0, 10.0]],
        query_ids=["q1"],
        query_vectors=[[3.0, 4.0]],
        k=2,
    )

    # Act
    l2_rows = compute_exact_ground_truth(**common_kwargs)  # type: ignore[arg-type]
    cosine_rows = compute_exact_ground_truth(**common_kwargs, metric="cosine")  # type: ignore[arg-type]

    # Assert
    assert [row.neighbor_id for row in l2_rows] == ["a", "b"]
    assert [row.neighbor_id for row in cosine_rows] == ["b", "a"]
    # Reported cosine distance is 1 - similarity, best rank closest to 0.
    assert cosine_rows[0].distance == pytest.approx(0.2)
    assert cosine_rows[1].distance == pytest.approx(0.4)


@pytest.mark.unit
def test_cosine_is_scale_invariant() -> None:
    # Arrange: scaling a base vector must not change cosine ranking.
    base_ids = ["a", "b"]
    base_vectors = [[3.0, 4.0], [1.0, 0.0]]
    scaled_base_vectors = [[30.0, 40.0], [0.5, 0.0]]

    # Act
    first = compute_exact_ground_truth(
        base_ids, base_vectors, ["q"], [[3.0, 4.0]], k=2, metric="cosine"
    )
    second = compute_exact_ground_truth(
        base_ids, scaled_base_vectors, ["q"], [[3.0, 4.0]], k=2, metric="cosine"
    )

    # Assert
    assert [row.neighbor_id for row in first] == [row.neighbor_id for row in second]
    assert first[0].neighbor_id == "a"


@pytest.mark.unit
def test_zero_vectors_produce_finite_deterministic_ground_truth() -> None:
    # Arrange: a zero base vector normalizes to zero — its cosine distance
    # to any query is exactly 1 (finite, no NaNs); a nonzero query ranks
    # the collinear axis vector strictly first.
    rows = compute_exact_ground_truth(
        base_ids=["zero", "axis"],
        base_vectors=[[0.0, 0.0], [2.0, 0.0]],
        query_ids=["q"],
        query_vectors=[[5.0, 0.0]],
        k=2,
        metric="cosine",
    )

    # Assert
    assert all(np.isfinite(row.distance) for row in rows)
    assert rows[0].neighbor_id == "axis"
    assert rows[0].distance == pytest.approx(0.0)
    assert rows[1].neighbor_id == "zero"
    assert rows[1].distance == pytest.approx(1.0)

    # A zero query ties with everything at distance 1; ordering must
    # still be deterministic (stable insertion-order tie-break).
    tie_kwargs: dict[str, object] = dict(
        base_ids=["zero", "axis"],
        base_vectors=[[0.0, 0.0], [2.0, 0.0]],
        query_ids=["q"],
        query_vectors=[[0.0, 0.0]],
        k=2,
        metric="cosine",
    )
    first = compute_exact_ground_truth(**tie_kwargs)  # type: ignore[arg-type]
    second = compute_exact_ground_truth(**tie_kwargs)  # type: ignore[arg-type]
    assert all(row.distance == pytest.approx(1.0) for row in first)
    assert [row.neighbor_id for row in first] == [row.neighbor_id for row in second]


@pytest.mark.unit
def test_default_metric_is_l2_for_backward_compatibility() -> None:
    # Arrange + Act: omitting metric must reproduce the historical L2
    # ranking (squared euclidean), not cosine.
    rows = compute_exact_ground_truth(
        base_ids=["a", "b"],
        base_vectors=[[1.0, 0.0], [0.0, 10.0]],
        query_ids=["q"],
        query_vectors=[[3.0, 4.0]],
        k=2,
    )

    # Assert
    assert [row.neighbor_id for row in rows] == ["a", "b"]
    assert rows[0].distance == pytest.approx(20.0)


@pytest.mark.unit
def test_unsupported_metric_rejected() -> None:
    # Act + Assert
    with pytest.raises(ValueError, match="Unsupported ground-truth metric"):
        compute_exact_ground_truth(
            base_ids=["a"],
            base_vectors=[[1.0, 0.0]],
            query_ids=["q"],
            query_vectors=[[1.0, 0.0]],
            k=1,
            metric="manhattan",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_runner_forwards_configured_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: capture the metric the runner forwards for a cosine config.
    runner = build_runner(tmp_path)
    runner.config.engines.distance_metric = "cosine"
    captured: dict[str, object] = {}

    def fake_compute(
        base_ids: list[str],
        base_vectors: list[list[float]],
        query_ids: list[str],
        query_vectors: list[list[float]],
        *,
        k: int,
        metric: str = "l2",
    ) -> list[GroundTruthRow]:
        captured["metric"] = metric
        return [
            GroundTruthRow(
                query_id=query_ids[0], rank=1, neighbor_id=base_ids[0], distance=0.0
            )
        ]

    monkeypatch.setattr(
        "benchvec.runner.compute_exact_ground_truth", fake_compute
    )
    records = make_prepared_records()

    # Act
    runner.ground_truth_for_queries(records, records[:2], k=2)

    # Assert
    assert captured["metric"] == "cosine"


@pytest.mark.unit
def test_cosine_ranking_kills_magnitude_only_ordering() -> None:
    # Arrange: raw inner products rank a=[10,0] first (IP 10 > 2), but
    # cosine ranks b=[0,1] first (cos 0.894 > 0.447) — a dropped or
    # query-only normalization shows up at the ranking level here.
    rows = compute_exact_ground_truth(
        base_ids=["a", "b"],
        base_vectors=[[10.0, 0.0], [0.0, 1.0]],
        query_ids=["q"],
        query_vectors=[[1.0, 2.0]],
        k=2,
        metric="cosine",
    )

    # Assert
    assert [row.neighbor_id for row in rows] == ["b", "a"]
