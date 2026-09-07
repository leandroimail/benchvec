"""Persistent LanceDB adapter for BenchVec workloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

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

logger = logging.getLogger(__name__)


def _metric_name(distance_metric: DistanceMetric) -> str:
    return "cosine" if distance_metric == "cosine" else "L2"

# Recommended maximum batch size for LanceDB .add() calls.
# LanceDB has no hard limit but large batches can exhaust memory or cause
# slow compaction; chunking keeps memory usage predictable.
_LANCEDB_MAX_BATCH_SIZE: int = 10_000

# PQ training with num_bits=8 needs 2**8 = 256 centroid rows; IVF kmeans
# needs at least one row per partition. Both are hard library requirements.
_PQ_MIN_TRAINING_ROWS: int = 256

class LanceDbEngine(BenchmarkEngine):
    """Adapter around embedded LanceDB tables.

    ``index_type`` is the *configured* intent (``"none"`` or ``"ivfpq"``);
    ``describe_runtime().index_type`` reports the *effective* mode derived
    from actual index existence (``"ivfpq"`` or ``"scan"``).
    """

    def __init__(
        self,
        storage_path: str | Path,
        *,
        table_name: str = "benchvec",
        distance_metric: DistanceMetric = "l2",
        index_type: Literal["none", "ivfpq"] = "none",
        num_partitions: int = 8,
        num_sub_vectors: int = 16,
        nprobes: int = 8,
        refine_factor: int = 1,
    ) -> None:
        self.name = "lancedb"
        self.distance_metric = distance_metric
        self.index_type = index_type
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.nprobes = nprobes
        self.refine_factor = refine_factor
        self.native_filtering = True
        self.storage_path = Path(storage_path)
        self.table_name = table_name
        self.last_search_params: dict[str, Any] = {}
        self._effective_index_type = "scan"
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
        self._refresh_effective_index_type()

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
        """Create the IVF_PQ index when configured; track the effective mode."""

        if self.index_type != "ivfpq":
            self._effective_index_type = "scan"
            return
        if self._table is None:
            self.connect()
        if self._table is None:
            return

        row_count = int(self._table.count_rows())
        hard_floor = max(_PQ_MIN_TRAINING_ROWS, self.num_partitions)
        if row_count < hard_floor:
            logger.warning(
                "LanceDB IVF_PQ skipped: %d rows below training floor %d "
                "(needs >= max(256 PQ rows, num_partitions)); running scan",
                row_count,
                hard_floor,
            )
            self._effective_index_type = "scan"
            return
        if row_count < self.num_partitions * _PQ_MIN_TRAINING_ROWS:
            logger.warning(
                "LanceDB IVF_PQ quantizer sampling is suboptimal: %d rows "
                "< num_partitions * 256 = %d; index quality may degrade",
                row_count,
                self.num_partitions * _PQ_MIN_TRAINING_ROWS,
            )

        dimension = self._vector_dimension()
        if dimension is not None and dimension % self.num_sub_vectors != 0:
            valid_divisors = [d for d in range(1, dimension + 1) if dimension % d == 0]
            raise ValueError(
                f"num_sub_vectors={self.num_sub_vectors} must divide vector "
                f"dimension={dimension}; valid divisors: {valid_divisors}"
            )

        self._table.create_index(
            metric=_metric_name(self.distance_metric),
            num_partitions=self.num_partitions,
            num_sub_vectors=self.num_sub_vectors,
            vector_column_name="vector",
            index_type="IVF_PQ",
        )
        self._refresh_effective_index_type()

    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        return self._query(query, k, filters=None)

    def delete(self, ids: Sequence[str]) -> int:
        if self._table is None:
            self.connect()
        if self._table is None or not ids:
            return 0
        id_list = list(ids)
        for offset in range(0, len(id_list), _LANCEDB_MAX_BATCH_SIZE):
            chunk = id_list[offset : offset + _LANCEDB_MAX_BATCH_SIZE]
            # Single-quote SQL literals with '' doubling: json.dumps-style
            # double-quoted literals break the lance SQL parser on ids
            # containing quotes or backslashes.
            id_literals = ", ".join(
                "'" + record_id.replace("'", "''") + "'" for record_id in chunk
            )
            self._table.delete(f"id IN ({id_literals})")
        # LanceDB delete does not report how many rows matched.
        return len(id_list)

    def upsert(self, records: Sequence[EngineRecord]) -> None:
        if self._table is None:
            self.connect()
        if not records:
            return
        if self._table is None:
            self.insert(records)
            return
        rows = [self._record_to_row(record) for record in records]
        (
            self._table.merge_insert(on="id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

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
        if self._effective_index_type == "ivfpq":
            if hasattr(query_builder, "nprobes"):
                query_builder = query_builder.nprobes(self.nprobes)
            refine_applied = self.refine_factor > 1 and hasattr(query_builder, "refine_factor")
            if refine_applied:
                query_builder = query_builder.refine_factor(self.refine_factor)
            self.last_search_params = {
                "index_type": "ivfpq",
                "num_partitions": self.num_partitions,
                "nprobes": self.nprobes,
                "refine_factor": self.refine_factor if refine_applied else 0,
            }
        else:
            self.last_search_params = {"index_type": "scan"}
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
            index_type=self._effective_index_type,
            native_filtering=self.native_filtering,
            storage_path=str(self.storage_path),
        )

    def _refresh_effective_index_type(self) -> None:
        """Derive the effective index mode from actual index existence."""

        if self._table is None:
            self._effective_index_type = "scan"
            return
        try:
            indices = self._table.list_indices()
            self._effective_index_type = "ivfpq" if len(list(indices)) > 0 else "scan"
        except Exception:  # pragma: no cover - defensive for API drift
            self._effective_index_type = "scan"

    def _vector_dimension(self) -> int | None:
        """Read the vector dimension from the table schema, when available."""

        if self._table is None:
            return None
        try:
            field_type = self._table.schema.field("vector").type
            return int(field_type.list_size)
        except Exception:  # pragma: no cover - schema variations
            return None

    @staticmethod
    def _record_to_row(record: EngineRecord) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": record.id,
            "vector": record.embedding,
            "text": record.document,
        }
        row.update(record.metadata)
        return row
