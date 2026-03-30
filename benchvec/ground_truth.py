"""Exact ground-truth generation and caching for recall computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field


class GroundTruthRow(BaseModel):
    """Exact nearest-neighbor record for one query at one rank."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    rank: int = Field(ge=1)
    neighbor_id: str
    distance: float


def _as_float32_matrix(vectors: Sequence[Sequence[float]]) -> npt.NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2-D matrix of vectors")
    return matrix


def compute_exact_ground_truth(
    base_ids: Sequence[str],
    base_vectors: Sequence[Sequence[float]],
    query_ids: Sequence[str],
    query_vectors: Sequence[Sequence[float]],
    *,
    k: int,
) -> list[GroundTruthRow]:
    """Compute exact L2 nearest neighbors for a query batch."""

    if k <= 0:
        raise ValueError("k must be positive")

    indexed_base_vectors = _as_float32_matrix(base_vectors)
    indexed_query_vectors = _as_float32_matrix(query_vectors)
    if indexed_base_vectors.shape[1] != indexed_query_vectors.shape[1]:
        raise ValueError("Base and query vectors must share the same dimensionality")

    top_k = min(k, len(base_ids))
    rows: list[GroundTruthRow] = []
    for query_id, query_vector in zip(query_ids, indexed_query_vectors):
        deltas = indexed_base_vectors - query_vector
        distances = np.sum(deltas * deltas, axis=1)
        best_indices = np.argsort(distances)[:top_k]
        for rank, index in enumerate(best_indices, start=1):
            rows.append(
                GroundTruthRow(
                    query_id=query_id,
                    rank=rank,
                    neighbor_id=base_ids[int(index)],
                    distance=float(distances[int(index)]),
                )
            )
    return rows


def persist_ground_truth(
    output_dir: str | Path,
    rows: Sequence[GroundTruthRow],
    *,
    dataset_name: str,
    k: int,
) -> None:
    """Persist ground truth rows and a manifest locally."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([row.model_dump(mode="json") for row in rows])
    pq.write_table(table, target_dir / "ground_truth.parquet")
    manifest = {"dataset_name": dataset_name, "k": k, "row_count": len(rows)}
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_ground_truth(output_dir: str | Path) -> list[GroundTruthRow]:
    """Load cached ground truth rows from disk."""

    target_dir = Path(output_dir)
    table = pq.read_table(target_dir / "ground_truth.parquet")
    return [
        GroundTruthRow.model_validate(row)
        for row in cast(list[dict[str, Any]], table.to_pylist())
    ]


def load_ground_truth_as_map(output_dir: str | Path) -> dict[str, list[str]]:
    """Load cached ground truth grouped by query ID in rank order."""

    grouped_rows: dict[str, list[tuple[int, str]]] = {}
    for row in load_ground_truth(output_dir):
        grouped_rows.setdefault(row.query_id, []).append((row.rank, row.neighbor_id))

    result: dict[str, list[str]] = {}
    for query_id, query_rows in grouped_rows.items():
        sorted_rows = sorted(query_rows, key=lambda item: item[0])
        result[query_id] = [neighbor_id for _, neighbor_id in sorted_rows]
    return result
