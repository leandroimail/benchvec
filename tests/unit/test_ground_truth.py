"""Unit tests for exact ground-truth computation and caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.ground_truth import (
    compute_exact_ground_truth,
    load_ground_truth,
    load_ground_truth_as_map,
    persist_ground_truth,
)


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
