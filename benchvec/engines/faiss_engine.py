"""Persistent FAISS adapter for BenchVec workloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np
import numpy.typing as npt

from benchvec.engines.base import (
    BenchmarkEngine,
    DistanceMetric,
    EngineRecord,
    EngineRuntime,
    MetadataFilter,
    MetadataValue,
    SearchHit,
    directory_size_bytes,
    package_version,
)


def _as_matrix(vectors: list[list[float]]) -> npt.NDArray[np.float32]:
    return cast(npt.NDArray[np.float32], np.asarray(vectors, dtype=np.float32))


def _normalize_rows(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return cast(npt.NDArray[np.float32], vectors / norms)


class FaissEngine(BenchmarkEngine):
    """Adapter around FAISS flat and IVFFlat indexes."""

    def __init__(
        self,
        storage_path: str | Path,
        *,
        distance_metric: DistanceMetric = "l2",
        index_type: str = "flat",
        nlist: int = 32,
    ) -> None:
        self.name = "faiss"
        self.distance_metric = distance_metric
        self.index_type = index_type
        self.native_filtering = False
        self.storage_path = Path(storage_path)
        self.nlist = nlist
        self._index: Any | None = None
        self._vector_dimension: int | None = None
        self._ids: list[str] = []
        self._documents: dict[str, str | None] = {}
        self._metadata: dict[str, dict[str, MetadataValue]] = {}
        self._stored_vectors: list[list[float]] = []
        self._pending_records: list[EngineRecord] = []

    def connect(self) -> None:
        import faiss

        self.storage_path.mkdir(parents=True, exist_ok=True)
        index_path = self.storage_path / "index.faiss"
        state_path = self.storage_path / "state.json"
        vectors_path = self.storage_path / "vectors.npy"
        if index_path.exists() and state_path.exists() and vectors_path.exists():
            self._index = faiss.read_index(str(index_path))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self._vector_dimension = cast(int | None, state.get("vector_dimension"))
            self._ids = cast(list[str], state.get("ids", []))
            self._documents = cast(dict[str, str | None], state.get("documents", {}))
            self._metadata = cast(dict[str, dict[str, MetadataValue]], state.get("metadata", {}))
            self._stored_vectors = cast(list[list[float]], np.load(vectors_path).tolist())

    def reset(self) -> None:
        self._index = None
        self._vector_dimension = None
        self._ids = []
        self._documents = {}
        self._metadata = {}
        self._stored_vectors = []
        self._pending_records = []
        if self.storage_path.exists():
            for item in self.storage_path.iterdir():
                if item.is_file():
                    item.unlink()

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        records = list(batch)
        if not records:
            return

        if self._vector_dimension is None:
            self._vector_dimension = len(records[0].embedding)
        transformed_vectors = self._prepare_vectors([record.embedding for record in records])
        self._ids.extend(record.id for record in records)
        self._stored_vectors.extend(cast(list[list[float]], transformed_vectors.tolist()))
        for record in records:
            self._documents[record.id] = record.document
            self._metadata[record.id] = dict(record.metadata)

        if self.index_type == "flat":
            if self._index is None:
                self._index = self._create_index(self._vector_dimension)
            self._index.add(transformed_vectors)
        else:
            self._pending_records.extend(records)

    def build_index(self) -> None:
        if self.index_type == "flat":
            return
        if self._vector_dimension is None:
            return
        self._index = self._create_index(self._vector_dimension)
        pending_vectors = self._prepare_vectors(
            [record.embedding for record in self._pending_records]
        )
        if len(pending_vectors) == 0:
            return
        self._index.train(pending_vectors)
        self._index.add(pending_vectors)
        self._pending_records = []

    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        if self._index is None:
            self.connect()
        if self._index is None:
            return []
        query_matrix = self._prepare_vectors([query])
        distances, indices = self._index.search(query_matrix, k)
        return self._hits_from_indices(distances[0], indices[0])

    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        filtered_indices = [
            index
            for index, record_id in enumerate(self._ids)
            if filters.matches(self._metadata.get(record_id, {}))
        ]
        if not filtered_indices:
            return []

        filtered_vectors = _as_matrix([self._stored_vectors[index] for index in filtered_indices])
        query_matrix = self._prepare_vectors([query])
        if self.distance_metric == "cosine":
            scores = filtered_vectors @ query_matrix[0]
            ordered_indices = np.argsort(-scores)[:k]
            distances = scores[ordered_indices]
        else:
            deltas = filtered_vectors - query_matrix[0]
            distances_full = np.sum(deltas * deltas, axis=1)
            ordered_indices = np.argsort(distances_full)[:k]
            distances = distances_full[ordered_indices]

        hits: list[SearchHit] = []
        for local_index, score in zip(ordered_indices.tolist(), distances.tolist()):
            record_id = self._ids[filtered_indices[local_index]]
            hits.append(
                SearchHit(
                    id=record_id,
                    score=float(score),
                    document=self._documents.get(record_id),
                    metadata=dict(self._metadata.get(record_id, {})),
                )
            )
        return hits

    def persist(self) -> None:
        import faiss

        if self._index is None and self.index_type != "flat":
            self.build_index()
        if self._index is None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.storage_path / "index.faiss"))
        state = {
            "vector_dimension": self._vector_dimension,
            "ids": self._ids,
            "documents": self._documents,
            "metadata": self._metadata,
        }
        (self.storage_path / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        np.save(self.storage_path / "vectors.npy", _as_matrix(self._stored_vectors))

    def reopen(self) -> None:
        self._index = None
        self._vector_dimension = None
        self._ids = []
        self._documents = {}
        self._metadata = {}
        self._stored_vectors = []
        self._pending_records = []
        self.connect()

    def disk_size(self) -> int:
        return directory_size_bytes(self.storage_path)

    def describe_runtime(self) -> EngineRuntime:
        return EngineRuntime(
            engine=self.name,
            package_version=package_version("faiss-cpu"),
            distance_metric=self.distance_metric,
            index_type=self.index_type,
            native_filtering=self.native_filtering,
            storage_path=str(self.storage_path),
        )

    def _create_index(self, dimension: int) -> Any:
        import faiss

        metric_type = (
            faiss.METRIC_INNER_PRODUCT
            if self.distance_metric == "cosine"
            else faiss.METRIC_L2
        )
        if self.index_type == "ivfflat":
            quantizer = (
                faiss.IndexFlatIP(dimension)
                if self.distance_metric == "cosine"
                else faiss.IndexFlatL2(dimension)
            )
            return faiss.IndexIVFFlat(quantizer, dimension, self.nlist, metric_type)
        if self.distance_metric == "cosine":
            return faiss.IndexFlatIP(dimension)
        return faiss.IndexFlatL2(dimension)

    def _prepare_vectors(self, vectors: list[list[float]]) -> npt.NDArray[np.float32]:
        matrix = _as_matrix(vectors)
        if self.distance_metric == "cosine":
            return _normalize_rows(matrix)
        return matrix

    def _hits_from_indices(
        self,
        distances: npt.NDArray[np.float32],
        indices: npt.NDArray[np.int64],
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for score, index in zip(distances.tolist(), indices.tolist()):
            if index < 0 or index >= len(self._ids):
                continue
            record_id = self._ids[index]
            hits.append(
                SearchHit(
                    id=record_id,
                    score=float(score),
                    document=self._documents.get(record_id),
                    metadata=dict(self._metadata.get(record_id, {})),
                )
            )
        return hits
