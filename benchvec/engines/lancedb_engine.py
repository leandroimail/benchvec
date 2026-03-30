"""Persistent LanceDB adapter for BenchVec workloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, cast

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


def _metric_name(distance_metric: DistanceMetric) -> str:
    return "cosine" if distance_metric == "cosine" else "L2"

# Recommended maximum batch size for LanceDB .add() calls.
# LanceDB has no hard limit but large batches can exhaust memory or cause
# slow compaction; chunking keeps memory usage predictable.
_LANCEDB_MAX_BATCH_SIZE: int = 10_000

class LanceDbEngine(BenchmarkEngine):
    """Adapter around embedded LanceDB tables."""

    def __init__(
        self,
        storage_path: str | Path,
        *,
        table_name: str = "benchvec",
        distance_metric: DistanceMetric = "l2",
    ) -> None:
        self.name = "lancedb"
        self.distance_metric = distance_metric
        self.index_type = "lance"
        self.native_filtering = True
        self.storage_path = Path(storage_path)
        self.table_name = table_name
        self._database: Any | None = None
        self._table: Any | None = None

    def connect(self) -> None:
        import lancedb

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._database = lancedb.connect(str(self.storage_path))
        _tables_resp = self._database.list_tables()
        # LanceDB 0.30+ returns ListTablesResponse(tables=[...]) instead of list[str]
        _table_list: list[str] = (
            _tables_resp.tables
            if hasattr(_tables_resp, "tables")
            else cast(list[str], _tables_resp)
        )
        table_names = set(_table_list)
        if self.table_name in table_names:
            self._table = self._database.open_table(self.table_name)
        else:
            self._table = None

    def reset(self) -> None:
        if self._database is None:
            self.connect()
        assert self._database is not None
        _tables_resp = self._database.list_tables()
        _table_list = (
            _tables_resp.tables
            if hasattr(_tables_resp, "tables")
            else cast(list[str], _tables_resp)
        )
        table_names = set(_table_list)
        if self.table_name in table_names:
            self._database.drop_table(self.table_name)
        self._table = None

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        """Insert records, chunking into batches of at most _LANCEDB_MAX_BATCH_SIZE.

        LanceDB has no hard per-call limit, but large batches cause memory
        pressure and slow compaction.  Splitting into chunks keeps resource
        consumption predictable across datasets of any size.
        """
        if self._database is None:
            self.connect()
        assert self._database is not None

        records = list(batch)
        if not records:
            return

        for offset in range(0, len(records), _LANCEDB_MAX_BATCH_SIZE):
            chunk = records[offset : offset + _LANCEDB_MAX_BATCH_SIZE]
            rows = [self._record_to_row(record) for record in chunk]
            if self._table is None:
                self._table = self._database.create_table(
                    self.table_name, data=rows, mode="overwrite"
                )
            else:
                self._table.add(rows)

    def build_index(self) -> None:
        return None

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
        if self._table is None:
            self.connect()
        if self._table is None:
            return []

        query_builder = self._table.search(query).distance_type(_metric_name(self.distance_metric))
        where_clause = None if filters is None else filters.to_lancedb_where()
        if where_clause:
            query_builder = query_builder.where(where_clause, prefilter=True)
        rows = cast(list[dict[str, Any]], query_builder.limit(k).to_list())
        hits: list[SearchHit] = []
        for row in rows:
            metadata = {
                key: cast(MetadataValue, value)
                for key, value in row.items()
                if key not in {"id", "text", "vector", "_distance"}
            }
            hits.append(
                SearchHit(
                    id=str(row["id"]),
                    score=float(row.get("_distance", 0.0)),
                    document=cast(str | None, row.get("text")),
                    metadata=metadata,
                )
            )
        return hits

    def persist(self) -> None:
        return None

    def reopen(self) -> None:
        self._database = None
        self._table = None
        self.connect()

    def disk_size(self) -> int:
        return directory_size_bytes(self.storage_path)

    def describe_runtime(self) -> EngineRuntime:
        return EngineRuntime(
            engine=self.name,
            package_version=package_version("lancedb"),
            distance_metric=self.distance_metric,
            index_type=self.index_type,
            native_filtering=self.native_filtering,
            storage_path=str(self.storage_path),
        )

    @staticmethod
    def _record_to_row(record: EngineRecord) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": record.id,
            "vector": record.embedding,
            "text": record.document,
        }
        row.update(record.metadata)
        return row
