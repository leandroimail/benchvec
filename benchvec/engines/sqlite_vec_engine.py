"""Persistent sqlite-vec adapter for BenchVec workloads.

Design notes (probed against sqlite-vec 0.1.9 on macOS/arm64, Python 3.12):

- The extension must be loaded with ``conn.enable_load_extension(True)``
  before ``sqlite_vec.load(conn)`` (the helper does not enable it itself).
- KNN query shape: ``SELECT id, distance FROM t WHERE embedding MATCH :query
  AND k = :k``. ``distance`` is an implicit column only available in KNN
  queries; there is no scalar ``distance(a, b)`` function on 0.1.9.
- ``distance_metric=cosine`` is a *column* option (no comma before it) and
  returns cosine *distance* (``1 - cosine_similarity``). L2 returns true
  euclidean distance (sqrt), not squared.
- vec0 metadata columns exist on 0.1.9 and support native pre-filtered KNN,
  BUT they reject NULL values and omitted columns for every declared
  metadata column. Our ``EngineRecord`` contract has optional documents and
  sparse metadata dictionaries, so faithful storage requires NULL tolerance:
  documents and metadata are stored in a companion regular SQLite table as
  (id, document, metadata JSON) and filtering is applied at the application
  layer, mirroring ``FaissEngine.search_filtered``. Hence
  ``native_filtering = False`` (capability probe evidence above).
- ``INSERT OR REPLACE`` fails on vec0 text primary keys ("UNIQUE constraint
  failed"), so upsert is implemented as DELETE + INSERT.
- Inserting or querying with a mismatched dimension raises
  ``sqlite3.OperationalError`` ("Dimension mismatch ..."); the adapter lets
  that native error propagate.
"""

from __future__ import annotations

import json
import logging
import threading
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

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

logger = logging.getLogger(__name__)

# Rows per executemany chunk; keeps per-statement memory predictable.
_INSERT_BATCH_SIZE: int = 512

# Ids per "IN (...)" clause; stays below SQLITE_MAX_VARIABLE_NUMBER (>=999).
_ID_CHUNK_SIZE: int = 500

# Extracts the dimension from the stored CREATE VIRTUAL TABLE statement.
_EMBEDDING_DIM_PATTERN = re.compile(r"embedding\s+float\[(\d+)\]")


def sqlite_vec_available() -> bool:
    """Return ``True`` when the sqlite-vec extension loads on this machine.

    Used by tests (skipif) and callers to degrade gracefully on platforms
    where SQLite extension loading is unavailable.
    """

    try:
        connection = sqlite3.connect(":memory:")
        try:
            _load_sqlite_vec(connection)
            return True
        finally:
            connection.close()
    except Exception:  # pragma: no cover - depends on host SQLite build
        return False


def _load_sqlite_vec(connection: sqlite3.Connection) -> None:
    """Load the bundled sqlite-vec extension into ``connection``.

    SQLite requires an explicit authorization switch before loading
    extensions; it is disabled again right after loading.
    """

    import sqlite_vec

    if not hasattr(connection, "enable_load_extension"):
        raise RuntimeError(
            "This Python sqlite3 build does not support loading extensions; "
            "sqlite-vec cannot be used."
        )
    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)


def _as_float32_bytes(vector: list[float]) -> bytes:
    """Serialize one embedding into sqlite-vec's compact little-endian f32 format."""

    return np.asarray(vector, dtype="<f4").tobytes()


def _as_matrix(vectors: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
    """Stack 1-D float32 vectors into a 2-D float32 matrix."""

    return np.vstack(vectors).astype(np.float32, copy=False)


def _normalize_rows(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalize rows, mapping zero vectors to zero (no NaNs)."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SqliteVecEngine(BenchmarkEngine):
    """Adapter around an embedded sqlite-vec ``vec0`` virtual table.

    ``vec0`` performs exhaustive (brute-force) KNN scans; there is no ANN
    structure to build, which is precisely this engine's benchmark design
    point, so ``build_index`` is a no-op and ``describe_runtime`` reports
    ``index_type="bruteforce-sqlite"``.

    All statement execution is serialized through an internal lock: the
    benchmark phases legitimately hop threads (timeout wrappers), and
    sqlite3 connections are not thread-safe on their own.
    """

    def __init__(
        self,
        storage_path: str | Path,
        *,
        distance_metric: DistanceMetric = "l2",
        table_name: str = "benchvec",
    ) -> None:
        self.name = "sqlite_vec"
        self.distance_metric = distance_metric
        # vec0 always scans exhaustively: brute force is its design point.
        self.index_type = "bruteforce-sqlite"
        # Probed on sqlite-vec 0.1.9: vec0 metadata columns support native
        # pre-filtered KNN but reject NULL/omitted values, which our
        # optional-document and sparse-metadata record contract requires.
        # We therefore store metadata as JSON in a companion table and
        # filter at the application layer (see class docstring).
        self.native_filtering = False
        self.storage_path = Path(storage_path)
        self.table_name = table_name
        self.last_search_params: dict[str, Any] = {}
        self._connection: sqlite3.Connection | None = None
        # Serializes all statement execution; sqlite3 connections are not
        # thread-safe even with check_same_thread=False.
        self._lock = threading.RLock()  # reentrant: upsert delegates to insert
        self._vector_dimension: int | None = None
        # True once the vec0 + companion tables exist in the database file.
        self._tables_ready: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the SQLite database and load the sqlite-vec extension.

        Restores the vector dimension from the persisted schema when the
        table already exists, so later inserts and cosine behavior stay
        consistent across reopen cycles.
        """

        if self._connection is not None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self._db_path), check_same_thread=False)
        _load_sqlite_vec(connection)
        self._connection = connection
        self._restore_schema_state()

    def reset(self) -> None:
        """Drop the vector and companion tables for a clean benchmark run."""

        if self._connection is None:
            self.connect()
        assert self._connection is not None
        with self._lock:
            self._connection.execute(f"DROP TABLE IF EXISTS {self._vec_table_name}")
            self._connection.execute(f"DROP TABLE IF EXISTS {self._records_table_name}")
            self._connection.commit()
        self._vector_dimension = None
        self._tables_ready = False

    def persist(self) -> None:
        """Commit pending writes to the database file."""

        if self._connection is not None:
            with self._lock:
                self._connection.commit()

    def reopen(self) -> None:
        """Close and reopen the persisted database state."""

        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._vector_dimension = None
        self._tables_ready = False
        self.connect()

    def disk_size(self) -> int:
        """Return the persisted database footprint in bytes."""

        return directory_size_bytes(self.storage_path)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def insert(self, batch: Iterable[EngineRecord]) -> None:
        """Insert records; embeddings are stored as compact f32 binary blobs.

        The ``vec0`` virtual table is created lazily on the first insert,
        once the vector dimension is known. Documents and metadata go into
        a companion regular table (metadata serialized as JSON) because
        vec0 metadata columns cannot store NULLs on 0.1.9.
        """

        records = list(batch)
        if not records:
            return
        if self._connection is None:
            self.connect()
        assert self._connection is not None

        with self._lock:
            # Step 1: lazily create the schema on the first insert.
            if not self._tables_ready:
                self._vector_dimension = len(records[0].embedding)
                self._create_tables()
                self._connection.commit()
                self._tables_ready = True

            # Step 2: chunked batched inserts into the vec0 table (binary blobs).
            for offset in range(0, len(records), _INSERT_BATCH_SIZE):
                chunk = records[offset : offset + _INSERT_BATCH_SIZE]
                vector_rows = [
                    (record.id, _as_float32_bytes(record.embedding)) for record in chunk
                ]
                self._connection.executemany(
                    f"INSERT INTO {self._vec_table_name}(id, embedding) VALUES (?, ?)",
                    vector_rows,
                )

            # Step 3: side data (document + metadata JSON) via upsert semantics;
            # plain SQLite tables natively support INSERT OR REPLACE.
            for offset in range(0, len(records), _INSERT_BATCH_SIZE):
                chunk = records[offset : offset + _INSERT_BATCH_SIZE]
                side_rows = [
                    (
                        record.id,
                        record.document,
                        json.dumps(record.metadata, sort_keys=True),
                    )
                    for record in chunk
                ]
                self._connection.executemany(
                    f"INSERT OR REPLACE INTO {self._records_table_name}"
                    "(id, document, metadata) VALUES (?, ?, ?)",
                    side_rows,
                )

    def build_index(self) -> None:
        """No-op by design: vec0 brute-force scans need no index build."""

        return None

    def upsert(self, records: Sequence[EngineRecord]) -> None:
        """Insert records, overwriting any with the same id.

        vec0 rejects ``INSERT OR REPLACE`` on text primary keys (probed on
        0.1.9), so existing ids are deleted first, then reinserted.
        """

        record_list = list(records)
        if not record_list:
            return
        if self._connection is None:
            self.connect()
        if not self._tables_ready:
            # Fresh database: nothing to overwrite, plain insert path.
            self.insert(record_list)
            return
        with self._lock:
            self._delete_ids([record.id for record in record_list])
            self.insert(record_list)

    def delete(self, ids: Sequence[str]) -> int:
        """Remove records by id; return the number actually removed."""

        id_list = list(ids)
        if not id_list:
            return 0
        if self._connection is None:
            self.connect()
        if not self._tables_ready:
            return 0
        return self._delete_ids(id_list)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Execute an unfiltered brute-force KNN search.

        Uses the native vec0 KNN form::

            SELECT id, distance FROM t
            WHERE embedding MATCH :query AND k = :k
            ORDER BY distance

        ``distance`` is an implicit column of KNN queries; the metric is
        fixed by the table's ``distance_metric`` at creation time, so the
        ``metric`` argument is intentionally ignored.
        """

        del metric, params
        if self._connection is None:
            self.connect()
        self.last_search_params = {"index_type": self.index_type}
        if not self._tables_ready:
            return []
        assert self._connection is not None

        with self._lock:
            # Step 1: native brute-force KNN over the vec0 table.
            cursor = self._connection.execute(
                f"SELECT id, distance FROM {self._vec_table_name} "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (_as_float32_bytes(query), k),
            )
            knn_rows: list[tuple[str, float]] = [
                (str(row[0]), float(row[1])) for row in cursor.fetchall()
            ]
            if not knn_rows:
                return []

            # Step 2: enrich hits with document + metadata from the side table.
            side_data = self._fetch_side_data([row_id for row_id, _ in knn_rows])
            return [
                self._hit_for(row_id, distance, side_data)
                for row_id, distance in knn_rows
            ]

    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Execute a metadata-filtered vector search.

        Application-layer pre-filter mirroring ``FaissEngine.search_filtered``:
        candidate ids are selected by evaluating the filter against stored
        metadata, then an exact numpy brute-force pass over just those
        embeddings produces the top-k. Scores use the same convention as the
        native path (euclidean distance for L2, ``1 - cosine_similarity`` for
        cosine) so filtered and unfiltered results stay comparable.
        """

        del metric, params
        if self._connection is None:
            self.connect()
        self.last_search_params = {
            "index_type": self.index_type,
            "filter_mode": "application",
        }
        if not self._tables_ready:
            return []
        assert self._connection is not None

        with self._lock:
            # Step 1: resolve matching ids at the application layer.
            side_rows = self._connection.execute(
                f"SELECT id, metadata FROM {self._records_table_name}"
            ).fetchall()
            filtered_ids: list[str] = []
            metadata_by_id: dict[str, dict[str, MetadataValue]] = {}
            for row_id, metadata_json in side_rows:
                metadata = json.loads(metadata_json)
                if filters.matches(metadata):
                    filtered_ids.append(str(row_id))
                    metadata_by_id[str(row_id)] = metadata
            if not filtered_ids:
                return []

            # Step 2: fetch only the candidate embeddings.
            candidates = self._fetch_embeddings(filtered_ids)
            if not candidates:
                return []
            ids = [row_id for row_id, _ in candidates]
            vectors = _as_matrix([embedding for _, embedding in candidates])
            query_vector = np.asarray([query], dtype="<f4")

            # Step 3: exact brute-force scoring consistent with the table metric.
            if self.distance_metric == "cosine":
                normalized_vectors = _normalize_rows(vectors)
                normalized_query = _normalize_rows(query_vector)
                similarity = normalized_vectors @ normalized_query[0]
                distances = 1.0 - similarity
            else:
                deltas = vectors - query_vector[0]
                distances = np.sqrt(np.sum(deltas * deltas, axis=1))

            # Step 4: ascending distance order, capped at k.
            order = np.argsort(distances)[:k]
            hits: list[SearchHit] = []
            for index in order.tolist():
                row_id = ids[index]
                hits.append(
                    SearchHit(
                        id=row_id,
                        score=float(distances[index]),
                        document=self._document_for(row_id),
                        metadata=dict(metadata_by_id.get(row_id, {})),
                    )
                )
            return hits

        # ------------------------------------------------------------------
        # Runtime description
        # ------------------------------------------------------------------
    def describe_runtime(self) -> EngineRuntime:
        """Describe the current runtime for reproducibility records."""

        return EngineRuntime(
            engine=self.name,
            package_version=package_version("sqlite-vec"),
            distance_metric=self.distance_metric,
            index_type=self.index_type,
            native_filtering=self.native_filtering,
            storage_path=str(self.storage_path),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def _db_path(self) -> Path:
        """Path of the SQLite database file backing this engine."""

        return self.storage_path / f"{self.table_name}.sqlite"

    @property
    def _vec_table_name(self) -> str:
        """Name of the vec0 virtual table (SQL identifiers are bare)."""

        return self.table_name

    @property
    def _records_table_name(self) -> str:
        """Name of the companion side-data table."""

        return f"{self.table_name}_records"

    def _create_tables(self) -> None:
        """Create the vec0 virtual table and the companion side-data table.

        The distance metric is fixed at creation via the vec0 column option
        (probed: ``embedding float[N] distance_metric=cosine|l2``); vec0
        column definitions must use bare (unquoted) identifiers.
        """

        assert self._connection is not None
        assert self._vector_dimension is not None
        metric_option = "distance_metric=cosine" if self.distance_metric == "cosine" else "distance_metric=l2"
        self._connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec_table_name} USING vec0("
            f"id text primary key, embedding float[{self._vector_dimension}] {metric_option})"
        )
        self._connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self._records_table_name}("
            "id TEXT PRIMARY KEY, document TEXT, metadata TEXT NOT NULL DEFAULT '{}')"
        )

    def _restore_schema_state(self) -> None:
        """Restore dimension/table readiness from an existing database file."""

        assert self._connection is not None
        row = self._connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (self._vec_table_name,),
        ).fetchone()
        if row is None or row[0] is None:
            self._tables_ready = False
            return
        self._tables_ready = True
        match = _EMBEDDING_DIM_PATTERN.search(str(row[0]))
        if match:
            self._vector_dimension = int(match.group(1))

    def _delete_ids(self, ids: Sequence[str]) -> int:
        """Delete rows by id from both tables; return the vec0 rowcount."""

        assert self._connection is not None
        removed = 0
        id_list = list(ids)
        for offset in range(0, len(id_list), _ID_CHUNK_SIZE):
            chunk = id_list[offset : offset + _ID_CHUNK_SIZE]
            placeholders = ", ".join("?" for _ in chunk)
            cursor = self._connection.execute(
                f"DELETE FROM {self._vec_table_name} WHERE id IN ({placeholders})",
                chunk,
            )
            removed += cursor.rowcount
            self._connection.execute(
                f"DELETE FROM {self._records_table_name} WHERE id IN ({placeholders})",
                chunk,
            )
        return removed

    def _fetch_side_data(self, ids: Sequence[str]) -> dict[str, tuple[str | None, dict[str, MetadataValue]]]:
        """Load (document, metadata) pairs for the given ids."""

        assert self._connection is not None
        side_data: dict[str, tuple[str | None, dict[str, MetadataValue]]] = {}
        id_list = list(ids)
        for offset in range(0, len(id_list), _ID_CHUNK_SIZE):
            chunk = id_list[offset : offset + _ID_CHUNK_SIZE]
            placeholders = ", ".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT id, document, metadata FROM {self._records_table_name} "
                f"WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row_id, document, metadata_json in rows:
                side_data[str(row_id)] = (
                    cast(str | None, document),
                    json.loads(metadata_json),
                )
        return side_data

    def _fetch_embeddings(self, ids: Sequence[str]) -> list[tuple[str, npt.NDArray[np.float32]]]:
        """Load candidate embeddings as float32 arrays for the given ids."""

        assert self._connection is not None
        candidates: list[tuple[str, npt.NDArray[np.float32]]] = []
        id_list = list(ids)
        for offset in range(0, len(id_list), _ID_CHUNK_SIZE):
            chunk = id_list[offset : offset + _ID_CHUNK_SIZE]
            placeholders = ", ".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT id, embedding FROM {self._vec_table_name} "
                f"WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row_id, blob in rows:
                candidates.append(
                    (str(row_id), np.frombuffer(blob, dtype="<f4"))
                )
        return candidates

    def _document_for(self, row_id: str) -> str | None:
        """Fetch just the document for one id from the side table."""

        assert self._connection is not None
        row = self._connection.execute(
            f"SELECT document FROM {self._records_table_name} WHERE id = ?",
            (row_id,),
        ).fetchone()
        return cast(str | None, row[0]) if row else None

    def _hit_for(
        self,
        row_id: str,
        distance: float,
        side_data: Mapping[str, tuple[str | None, dict[str, MetadataValue]]],
    ) -> SearchHit:
        """Build a ``SearchHit`` from a KNN row plus its side data."""

        document, metadata = side_data.get(row_id, (None, {}))
        return SearchHit(
            id=row_id,
            score=distance,
            document=document,
            metadata=dict(metadata),
        )
