"""TurboVec adapter for BenchVec workloads.

Design notes (probed against turbovec 1.0.0 on macOS/arm64, Python 3.12):

- TurboVec is a Rust vector index (PyO3 bindings) built on Google's
  TurboQuant — a data-oblivious quantizer, so there is no train step:
  ``build_index`` only runs the optional TQ+ ``calibrate`` (≤1024-row
  sample) and ``prepare``; the engine's design point is exhaustive scan
  with heavy scalar compression (``index_type="turboquant-<bits>bit"``).
- The kernel is **inner product only** (probed: 10·e₁ ranks above e₁).
  The adapter therefore supports ``cosine`` (L2-normalize vectors and
  queries; IP ordering on unit vectors equals cosine ordering) and
  rejects ``l2`` with a clear error both at construction and per call.
- ``dim`` must be a positive multiple of 8; inputs must be 2-D float32
  (probed: 1-D and float64 raise TypeError). The adapter converts.
- ``bit_width`` must be 2, 3, or 4 (probed: 8 raises ValueError).
- ``IdMapIndex`` backs stable external uint64 ids: duplicate id on add
  raises ValueError (upsert = remove + add), ``remove`` of a missing id
  is a no-op, ``contains(id)`` exists, and ``k`` greater than the index
  size returns ``min(k, n)`` rows without error.
- Search accepts an id **allowlist** evaluated inside the SIMD kernel;
  an *empty* allowlist raises ValueError, so the adapter short-circuits
  to ``[]`` before calling the library. General metadata filtering does
  not exist in the library: the ``MetadataFilter`` itself is evaluated
  at the application layer against a JSON sidecar, hence
  ``native_filtering = False`` (mirrors ``FaissEngine.search_filtered``).
- Thread safety (probed): 8 threads × 200 concurrent searches alongside
  concurrent ``remove``/``add_with_ids`` completed without crashes,
  garbage ids, or native errors — the Rust side synchronizes. The
  adapter therefore locks (RLock) only id-map and index *mutations* and
  runs searches lock-free; a search racing a delete skips tombstoned
  (``None``) id-map slots.
- Documents are not stored (metrics never read ``SearchHit.document``);
  metadata is kept in a JSON sidecar solely to evaluate filters, so
  filtered hits can still be produced. Persisted footprint
  (``index.tv`` + two JSON sidecars) is what ``disk_size`` reports.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

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

if TYPE_CHECKING:
    import turbovec

logger = logging.getLogger(__name__)

# Library-legal quantization bit widths (probed: anything else raises).
_LEGAL_BIT_WIDTHS: tuple[int, ...] = (2, 3, 4)

# Rows per add_with_ids chunk (library's own BATCH_CHUNK_SIZE default).
_INSERT_CHUNK_SIZE: int = 4096

# TQ+ calibration sample cap (README recommends ~1024 rows).
_CALIBRATION_SAMPLE_ROWS: int = 1024


def turbovec_available() -> bool:
    """Return ``True`` when the turbovec package is importable.

    Used by tests (skipif) and callers to degrade gracefully on platforms
    without a prebuilt wheel (musl / old-glibc hosts would need a local
    Rust build, which BenchVec never performs).
    """

    return find_spec("turbovec") is not None


def _normalize_rows(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalize rows, mapping zero vectors to zero (no NaNs)."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically (temp file + ``os.replace``)."""

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp_path, path)


class TurboVecEngine(BenchmarkEngine):
    """Adapter around a turbovec ``IdMapIndex`` (TurboQuant exhaustive scan).

    Ids: dense uint64 assignment on insert with **tombstone-on-delete**
    (``_ids_by_uint[uint] = None``; the next uint is always
    ``len(_ids_by_uint)`` — slots are never reused or shifted, so the
    uint→string mapping stays stable for the index's lifetime).
    """

    def __init__(
        self,
        storage_path: str | Path,
        *,
        distance_metric: DistanceMetric = "cosine",
        bit_width: int = 4,
        calibrate: bool = False,
    ) -> None:
        if distance_metric != "cosine":
            raise ValueError(
                "TurboVecEngine supports distance_metric='cosine' only "
                "(the turbovec kernel is inner-product-only); configure "
                "engines.distance_metric='cosine'"
            )
        if bit_width not in _LEGAL_BIT_WIDTHS:
            raise ValueError(
                f"bit_width must be one of {_LEGAL_BIT_WIDTHS}, got {bit_width}"
            )

        self.name = "turbovec"
        self.distance_metric: DistanceMetric = distance_metric
        self.bit_width = bit_width
        # Data-oblivious quantizer: exhaustive scan is the design point.
        self.index_type = f"turboquant-{bit_width}bit"
        # MetadataFilter evaluation happens outside the engine; only the
        # resulting id allowlist is pushed into the SIMD kernel.
        self.native_filtering = False
        self.storage_path = Path(storage_path)
        self.last_search_params: dict[str, Any] = {}
        # Reentrant: upsert delegates to insert under the same lock.
        self._lock = threading.RLock()
        self._index: turbovec.IdMapIndex | None = None
        self._dim: int | None = None
        self._id_to_uint: dict[str, int] = {}
        self._ids_by_uint: list[str | None] = []
        self._metadata_by_id: dict[str, dict[str, MetadataValue]] = {}
        # TQ+ calibration sample; retained after calibration so sweep
        # rebuilds (new bit width) can re-calibrate from it.
        self._calibration_buffer: list[npt.NDArray[np.float32]] = []
        self._calibration_rows = 0
        self.calibrate_enabled = calibrate

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Ensure storage exists; restore persisted state when present."""

        import turbovec

        self.storage_path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            return
        if self._index_path.exists():
            self._index = turbovec.IdMapIndex.load(str(self._index_path))
            self._dim = int(self._index.dim)
        if self._ids_path.exists():
            try:
                payload = json.loads(self._ids_path.read_text(encoding="utf-8"))
                ids_list = payload.get("ids") if isinstance(payload, dict) else None
                if not isinstance(ids_list, list) or not all(
                    entry is None or isinstance(entry, str) for entry in ids_list
                ):
                    raise ValueError("ids sidecar has an unexpected shape")
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                # An IdMapIndex is unusable without its id map: re-adding
                # records would collide with uints the loaded snapshot
                # already owns. Discard the snapshot and start fresh.
                logger.warning(
                    "corrupt ids sidecar at %s; discarding the index "
                    "snapshot as well (fresh engine state)",
                    self._ids_path,
                )
                self._index = None
                self._dim = None
                self._ids_by_uint = []
                self._id_to_uint = {}
            else:
                self._ids_by_uint = list(ids_list)
                self._id_to_uint = {
                    record_id: uint
                    for uint, record_id in enumerate(self._ids_by_uint)
                    if record_id is not None
                }
        if self._index is not None and not self._ids_path.exists():
            # Snapshot WITHOUT its id sidecar (crash between index.write
            # and the sidecar write) is the same unusable state as a
            # corrupt one: the map is the only uint→string translation,
            # so re-adding records would collide with snapshot uints.
            logger.warning(
                "index snapshot at %s has no ids sidecar; discarding the "
                "snapshot (fresh engine state)",
                self._index_path,
            )
            self._index = None
            self._dim = None
        if self._metadata_path.exists():
            try:
                payload = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not all(
                    isinstance(value, dict) for value in payload.values()
                ):
                    raise ValueError("metadata sidecar has an unexpected shape")
                self._metadata_by_id = payload
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                # Degrade to empty metadata (unfiltered search still works;
                # filters match nothing) instead of failing the reopen.
                logger.warning(
                    "corrupt metadata sidecar at %s; filters will match "
                    "nothing until records are re-inserted",
                    self._metadata_path,
                )
                self._metadata_by_id = {}

    def reset(self) -> None:
        """Delete persisted files and clear all in-memory state.

        ``_load_engine`` calls ``reset`` before every workload repetition;
        leftover files would make the next insert hit turbovec's
        duplicate-id ValueError from repetition 2 onward.
        """

        with self._lock:
            for path in (self._index_path, self._ids_path, self._metadata_path):
                path.unlink(missing_ok=True)
            self._index = None
            self._dim = None
            self._id_to_uint = {}
            self._ids_by_uint = []
            self._metadata_by_id = {}
            self._calibration_buffer = []
            self._calibration_rows = 0

    def persist(self) -> None:
        """Write the index snapshot and both JSON sidecars durably.

        The index is written first, sidecars after (atomic per file). A
        crash between leaves a fresh/absent map with a stale snapshot;
        reopen degrades safely (corrupt/missing ids sidecar discards the
        snapshot), at the cost of losing uncommitted records.
        """

        if self._index is None:
            return
        with self._lock:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._index.write(str(self._index_path), durable=True)
            _atomic_write_json(self._ids_path, {"ids": self._ids_by_uint})
            _atomic_write_json(self._metadata_path, self._metadata_by_id)

    def reopen(self) -> None:
        """Drop in-memory state and reload the persisted index + sidecars."""

        with self._lock:
            self._index = None
            self._dim = None
            self._id_to_uint = {}
            self._ids_by_uint = []
            self._metadata_by_id = {}
            # The calibration buffer is in-memory-only by design; a
            # reopened engine rebuilds it if more inserts arrive.
            self._calibration_buffer = []
            self._calibration_rows = 0
            self.connect()

    def disk_size(self) -> int:
        """Return the persisted footprint (index + sidecars) in bytes."""

        return directory_size_bytes(self.storage_path)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def insert(self, batch: Iterable[EngineRecord]) -> None:
        """Insert records; chunked ``add_with_ids`` under the id-map lock.

        Rejects duplicate string ids up front (in-batch or already
        stored) — a mid-chunk library ValueError would leave a partially
        extended id map.
        """

        records = list(batch)
        if not records:
            return
        import turbovec

        with self._lock:
            # Step 1: fail fast on duplicates and mixed dims before
            # touching the library — a mid-chunk library ValueError would
            # leave a partially extended id map.
            dim = len(records[0].embedding)
            seen: set[str] = set()
            for record in records:
                if record.id in seen or record.id in self._id_to_uint:
                    msg = (
                        f"record id {record.id!r} already present; "
                        "use upsert() to overwrite"
                    )
                    raise ValueError(msg)
                if len(record.embedding) != dim:
                    raise ValueError(
                        f"embedding dim {len(record.embedding)} does not "
                        f"match batch dim {dim}"
                    )
                seen.add(record.id)

            # Step 2: lazy index creation once the dimension is known.
            if self._dim is None:
                if dim % 8 != 0:
                    raise ValueError(
                        f"turbovec requires dim to be a multiple of 8, got {dim}"
                    )
                self._dim = dim
                self._index = turbovec.IdMapIndex(dim, self.bit_width)
            elif dim != self._dim:
                raise ValueError(
                    f"embedding dim {dim} does not match index dim {self._dim}"
                )
            assert self._index is not None

            # Step 3: chunked add_with_ids, extending the id map per chunk.
            for offset in range(0, len(records), _INSERT_CHUNK_SIZE):
                chunk = records[offset : offset + _INSERT_CHUNK_SIZE]
                vectors = self._prepare_vectors([r.embedding for r in chunk])
                uints = self._append_id_map([r.id for r in chunk])
                self._index.add_with_ids(vectors, uints)
                self._append_calibration_sample(vectors)
            for record in records:
                self._metadata_by_id[record.id] = dict(record.metadata)

    def build_index(self) -> None:
        """Run optional TQ+ calibration, then ``prepare``.

        The calibration sample is retained after calibration (not cleared)
        so a later ``rebuild_with_bit_width`` can re-calibrate from it.
        """

        with self._lock:
            if self._index is None:
                return
            if (
                self.calibrate_enabled
                and self._calibration_rows > 0
                and self._index.calibration_state == "uncalibrated"
            ):
                self._index.calibrate(np.vstack(self._calibration_buffer))
            self._index.prepare()

    def rebuild_with_bit_width(self, bit_width: int) -> None:
        """Swap the underlying index to a new quantization bit width.

        The sweep applies points live and never re-calls ``build_index``,
        so the caller must re-insert all records afterwards; the retained
        calibration sample lets the fresh index re-calibrate immediately
        when ``calibrate`` is enabled (never silently uncalibrated).
        """

        if bit_width not in _LEGAL_BIT_WIDTHS:
            raise ValueError(
                f"bit_width must be one of {_LEGAL_BIT_WIDTHS}, got {bit_width}"
            )
        import turbovec

        with self._lock:
            if self._dim is None:
                raise RuntimeError(
                    "rebuild_with_bit_width requires at least one insert "
                    "(the vector dimension is learned there)"
                )
            self._index = turbovec.IdMapIndex(self._dim, bit_width)
            self.bit_width = bit_width
            self.index_type = f"turboquant-{bit_width}bit"
            self._id_to_uint = {}
            self._ids_by_uint = []
            self._metadata_by_id = {}
            if (
                self.calibrate_enabled
                and self._calibration_rows > 0
                and self._index.calibration_state == "uncalibrated"
            ):
                self._index.calibrate(np.vstack(self._calibration_buffer))
            self._index.prepare()

    def upsert(self, records: Sequence[EngineRecord]) -> None:
        """Insert records, overwriting any with the same id.

        turbovec raises ValueError on duplicate ids, so existing ids are
        removed first (tombstoned in the map), then re-added with fresh
        uints via the normal insert path.
        """

        record_list = list(records)
        if not record_list:
            return
        with self._lock:
            self.delete([record.id for record in record_list])
            self.insert(record_list)

    def delete(self, ids: Sequence[str]) -> int:
        """Remove records by id; tombstone their map slots. Returns count."""

        id_list = list(ids)
        if not id_list:
            return 0
        with self._lock:
            removed = 0
            for record_id in id_list:
                uint = self._id_to_uint.pop(record_id, None)
                if uint is None:
                    # turbovec's remove of a missing id is a no-op; here
                    # the id was never present (or already deleted).
                    continue
                assert self._index is not None
                self._index.remove(uint)
                self._ids_by_uint[uint] = None
                self._metadata_by_id.pop(record_id, None)
                removed += 1
            return removed

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
        """Execute an unfiltered exhaustive TurboQuant search.

        Scores are inner products (higher = better); under cosine the
        vectors are unit-norm so this equals cosine similarity. Documents
        are not stored and metrics never read hit payloads beyond ids.
        """

        del params
        self._validate_metric(metric)
        # _index is only nulled by reset()/reopen(), which the runner
        # invokes sequentially between phases — never concurrently with
        # the lock-free searches below.
        if self._index is None or not self._id_to_uint:
            return []
        self.last_search_params = {
            "bit_width": self.bit_width,
            "calibration_state": self._index.calibration_state,
        }

        # Step 1: normalize the query under cosine, then one batched call.
        query_matrix = self._prepare_vectors([query])
        scores, uint_ids = self._index.search(query_matrix, k)

        # Step 2: map library uints back to string ids, skipping slots
        # tombstoned by a concurrent delete (searches run lock-free).
        hits: list[SearchHit] = []
        score_rows = np.asarray(scores).ravel()
        id_rows = np.asarray(uint_ids).ravel()
        for score, uint_id in zip(score_rows, id_rows):
            index = int(uint_id)
            if 0 <= index < len(self._ids_by_uint):
                record_id = self._ids_by_uint[index]
                if record_id is not None:
                    hits.append(SearchHit(id=record_id, score=float(score)))
        return hits

    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Execute a metadata-filtered search via the kernel allowlist.

        The ``MetadataFilter`` is evaluated at the application layer
        against the metadata sidecar; matching ids are passed to turbovec
        as a uint64 allowlist (evaluated inside the SIMD kernel). An empty
        match short-circuits to ``[]`` — the library raises ValueError on
        an empty allowlist (probed).
        """

        del params
        self._validate_metric(metric)
        # Same sequential-phases assumption as ``search`` (see there).
        if self._index is None or not self._id_to_uint:
            return []
        self.last_search_params = {
            "bit_width": self.bit_width,
            "calibration_state": self._index.calibration_state,
            "filter_mode": "allowlist",
        }

        # Step 1: resolve matching ids at the application layer (locked so
        # the id snapshot is consistent with the map being mutated).
        with self._lock:
            matching_ids = [
                record_id
                for record_id, metadata in self._metadata_by_id.items()
                if filters.matches(metadata)
            ]
            allowlist = [
                self._id_to_uint[record_id]
                for record_id in matching_ids
                if record_id in self._id_to_uint
            ]
        if not allowlist:
            return []

        # Step 2: single batched search restricted to the allowlist.
        query_matrix = self._prepare_vectors([query])
        scores, uint_ids = self._index.search(
            query_matrix, k, allowlist=np.asarray(allowlist, dtype=np.uint64)
        )
        hits: list[SearchHit] = []
        score_rows = np.asarray(scores).ravel()
        id_rows = np.asarray(uint_ids).ravel()
        for score, uint_id in zip(score_rows, id_rows):
            index = int(uint_id)
            if 0 <= index < len(self._ids_by_uint):
                record_id = self._ids_by_uint[index]
                if record_id is not None:
                    hits.append(SearchHit(id=record_id, score=float(score)))
        return hits

    # ------------------------------------------------------------------
    # Runtime description
    # ------------------------------------------------------------------
    def describe_runtime(self) -> EngineRuntime:
        """Describe the current runtime for reproducibility records."""

        return EngineRuntime(
            engine=self.name,
            package_version=package_version("turbovec"),
            distance_metric=self.distance_metric,
            index_type=self.index_type,
            native_filtering=self.native_filtering,
            storage_path=str(self.storage_path),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        """Path of the IdMapIndex snapshot file."""

        return self.storage_path / "index.tv"

    @property
    def _ids_path(self) -> Path:
        """Path of the uint→string id map sidecar."""

        return self.storage_path / "ids.json"

    @property
    def _metadata_path(self) -> Path:
        """Path of the metadata sidecar (filter evaluation only)."""

        return self.storage_path / "metadata.json"

    def _validate_metric(self, metric: DistanceMetric) -> None:
        """Reject metrics the turbovec kernel cannot evaluate."""

        if metric == "l2":
            raise ValueError(
                "turbovec evaluates inner product only; configure "
                "engines.distance_metric='cosine' (the kernel is "
                "inner-product-only; l2 is not supported)"
            )

    def _prepare_vectors(
        self, embeddings: Sequence[list[float]]
    ) -> npt.NDArray[np.float32]:
        """Convert embeddings to a 2-D float32 matrix (cosine-normalized).

        turbovec rejects anything but 2-D float32 (probed), so integer
        input lists are cast here; zero vectors stay zero (no NaNs).
        """

        matrix = np.asarray(embeddings, dtype=np.float32)
        if self.distance_metric == "cosine":
            matrix = _normalize_rows(matrix)
        return matrix

    def _append_id_map(self, record_ids: Sequence[str]) -> npt.NDArray[np.uint64]:
        """Assign the next dense uints to ``record_ids`` and return them."""

        uints = np.empty(len(record_ids), dtype=np.uint64)
        for offset, record_id in enumerate(record_ids):
            uint = len(self._ids_by_uint)
            self._ids_by_uint.append(record_id)
            self._id_to_uint[record_id] = uint
            uints[offset] = uint
        return uints

    def _append_calibration_sample(
        self, vectors: npt.NDArray[np.float32]
    ) -> None:
        """Retain up to ~1024 normalized rows for optional TQ+ calibration."""

        if not self.calibrate_enabled or self._calibration_rows >= _CALIBRATION_SAMPLE_ROWS:
            return
        remaining = _CALIBRATION_SAMPLE_ROWS - self._calibration_rows
        sample = vectors[:remaining]
        self._calibration_buffer.append(sample)
        self._calibration_rows += len(sample)
