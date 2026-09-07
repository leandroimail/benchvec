"""Persistent ChromaDB adapter for BenchVec workloads."""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

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

# Pattern that matches ISO-8601 dates like "2024-03-15"
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ChromaDB enforces a hard limit on batch size per .add() call
# (InternalError raised above this threshold in all versions up to 1.x)
_CHROMA_MAX_BATCH_SIZE: int = 5461

logger = logging.getLogger(__name__)


def _iso_date_to_unix(value: str) -> int:
    """Convert an ISO-8601 date string (YYYY-MM-DD) to a UTC Unix timestamp (int)."""
    parsed = dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def _coerce_chroma_value(value: MetadataValue) -> MetadataValue:
    """Convert ISO date strings to int timestamps so ChromaDB range operators work."""
    if isinstance(value, str) and _ISO_DATE_RE.match(value):
        return _iso_date_to_unix(value)
    return value


def _normalize_metadata_for_chroma(metadata: dict[str, MetadataValue]) -> dict[str, MetadataValue]:
    """Return metadata copy with ISO date strings converted to Unix int timestamps.

    ChromaDB 1.x rejects empty metadata dicts; a sentinel key is added when
    the source metadata has no entries so the collection.add() call succeeds.
    """
    result = {key: _coerce_chroma_value(value) for key, value in metadata.items()}
    if not result:
        # Sentinel to satisfy ChromaDB 1.x non-empty metadata requirement
        result["_bv"] = 1
    return result


def _build_chroma_where(filters: MetadataFilter) -> dict[str, Any] | None:
    """Build ChromaDB ``where`` dict, coercing ISO date strings to numeric timestamps."""
    if not filters.clauses:
        return None

    operator_map = {
        "eq": "$eq",
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
    }
    conditions: list[dict[str, Any]] = []
    for clause in filters.clauses:
        coerced = _coerce_chroma_value(clause.value)
        if clause.operator == "eq":
            conditions.append({clause.field: coerced})
        else:
            conditions.append({clause.field: {operator_map[clause.operator]: coerced}})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _normalize_vector(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return cast(list[float], array.tolist())
    return cast(list[float], (array / norm).tolist())


class ChromaEngine(BenchmarkEngine):
    """Adapter around ChromaDB PersistentClient collections.

    HNSW ``ef`` is fixed at collection creation: sweeping it requires a
    rebuild per grid point (``rebuild_with_ef``, used only by the gated
    Pareto sweep). ``last_search_params`` reports the ef applied at query
    time, which is always the creation-time value.
    """

    def __init__(
        self,
        storage_path: str | Path,
        *,
        collection_name: str = "benchvec",
        distance_metric: DistanceMetric = "l2",
        search_ef: int = 64,
        ef_search_grid_enabled: bool = False,
        ef_search_grid: list[int] | None = None,
    ) -> None:
        self.name = "chromadb"
        self.distance_metric = distance_metric
        self.index_type = "hnsw"
        self.native_filtering = True
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.search_ef = search_ef
        self.ef_search_grid_enabled = ef_search_grid_enabled
        self.ef_search_grid = list(ef_search_grid or [])
        self.last_search_params: dict[str, Any] = {}
        self.ef_creation_mode: str = "pending"
        self._client: Any | None = None
        self._collection: Any | None = None

    def connect(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()
        logger.info(
            "chromadb collection created with ef_search=%d via %s",
            self.search_ef,
            self.ef_creation_mode,
        )

    def _get_or_create_collection(self) -> Any:
        """Create the collection, preferring the chromadb 1.x configuration API.

        Falls back to legacy ``hnsw:*`` metadata when ``configuration=`` is
        rejected. The two mechanisms are never used simultaneously;
        construction ef stays at the library default.
        """

        assert self._client is not None
        space = "cosine" if self.distance_metric == "cosine" else "l2"
        try:
            collection = self._client.get_or_create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": space, "ef_search": self.search_ef}},
                embedding_function=None,
            )
            self.ef_creation_mode = "configuration"
            return collection
        except Exception as exc:
            # chromadb raises typed errors (InvalidArgumentError etc.) that do
            # not share a ValueError base; any failure here means the
            # configuration API is unavailable on this install.
            logger.debug("configuration= path rejected, using legacy metadata: %s", exc)
            collection = self._client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": space, "hnsw:search_ef": self.search_ef},
                embedding_function=None,
            )
            self.ef_creation_mode = "legacy-metadata"
            return collection

    def rebuild_with_ef(self, search_ef: int) -> None:
        """Drop and recreate the collection with a new ef (data must be re-ingested)."""

        self.search_ef = search_ef
        if self._client is None:
            self.connect()
        assert self._client is not None
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._get_or_create_collection()

    def reset(self) -> None:
        if self._client is None:
            self.connect()
        assert self._client is not None
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        self.connect()

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        """Insert records, chunking into batches of at most _CHROMA_MAX_BATCH_SIZE.

        ChromaDB enforces a hard limit of 5461 records per .add() call.  When
        the caller passes a larger slice the method splits it transparently and
        performs multiple sequential .add() calls until all records are stored.
        """
        if self._collection is None:
            self.connect()
        assert self._collection is not None

        records = list(batch)
        if not records:
            return

        # Chunk into sub-batches respecting the ChromaDB internal limit
        for offset in range(0, len(records), _CHROMA_MAX_BATCH_SIZE):
            chunk = records[offset : offset + _CHROMA_MAX_BATCH_SIZE]
            embeddings = [
                _normalize_vector(record.embedding)
                if self.distance_metric == "cosine"
                else record.embedding
                for record in chunk
            ]
            self._collection.add(
                ids=[record.id for record in chunk],
                embeddings=embeddings,
                documents=[record.document for record in chunk],
                metadatas=[
                    _normalize_metadata_for_chroma(record.metadata) for record in chunk
                ],
            )

    def build_index(self) -> None:
        return None

    def delete(self, ids: Sequence[str]) -> int:
        if self._collection is None:
            self.connect()
        assert self._collection is not None
        if not ids:
            return 0
        self._collection.delete(ids=list(ids))
        # ChromaDB delete does not report how many ids existed.
        return len(list(ids))

    def upsert(self, records: Sequence[EngineRecord]) -> None:
        if self._collection is None:
            self.connect()
        assert self._collection is not None
        if not records:
            return
        record_list = list(records)
        # The chromadb hard per-call limit applies to upsert as well.
        for offset in range(0, len(record_list), _CHROMA_MAX_BATCH_SIZE):
            chunk = record_list[offset : offset + _CHROMA_MAX_BATCH_SIZE]
            embeddings = [
                _normalize_vector(record.embedding)
                if self.distance_metric == "cosine"
                else record.embedding
                for record in chunk
            ]
            self._collection.upsert(
                ids=[record.id for record in chunk],
                embeddings=embeddings,
                documents=[record.document for record in chunk],
                metadatas=[
                    _normalize_metadata_for_chroma(record.metadata) for record in chunk
                ],
            )

    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        return self._query(query, k, filters=None)

    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        return self._query(query, k, filters=filters)

    def _query(
        self,
        query: list[float],
        k: int,
        *,
        filters: MetadataFilter | None,
    ) -> list[SearchHit]:
        if self._collection is None:
            self.connect()
        assert self._collection is not None

        query_vector = _normalize_vector(query) if self.distance_metric == "cosine" else query
        where_clause = _build_chroma_where(filters) if filters is not None else None
        self.last_search_params = {"ef_search": self.search_ef}
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where_clause,
            include=["metadatas", "documents", "distances"],
        )
        ids = cast(list[list[str]], result["ids"])
        metadatas = cast(list[list[dict[str, Any]]], result.get("metadatas") or [[]])
        documents = cast(list[list[str | None]], result.get("documents") or [[]])
        distances = cast(list[list[float]], result.get("distances") or [[]])
        hits: list[SearchHit] = []
        for index, record_id in enumerate(ids[0] if ids else []):
            hits.append(
                SearchHit(
                    id=record_id,
                    score=distances[0][index],
                    document=documents[0][index] if documents else None,
                    metadata=dict(metadatas[0][index]) if metadatas else {},
                )
            )
        return hits

    def persist(self) -> None:
        return None

    def reopen(self) -> None:
        self._client = None
        self._collection = None
        self.connect()

    def disk_size(self) -> int:
        return directory_size_bytes(self.storage_path)

    def describe_runtime(self) -> EngineRuntime:
        return EngineRuntime(
            engine=self.name,
            package_version=package_version("chromadb"),
            distance_metric=self.distance_metric,
            index_type=self.index_type,
            native_filtering=self.native_filtering,
            storage_path=str(self.storage_path),
        )
