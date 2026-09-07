"""Parquet-backed raw measurement storage for benchmark micro-data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field


class RawRecord(BaseModel):
    """Single raw measurement record aligned with PRD03 query/ingest schemas."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    engine: str
    workload: str
    phase: str
    status: str
    timestamp_utc: str
    repetition_index: int = Field(default=0, ge=0)

    query_id: str | None = None
    query_index: int | None = Field(default=None, ge=0)
    mode: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    returned_ids: list[str] | None = None
    ground_truth_ids: list[str] | None = None
    filter_applied: str | None = None
    filter_selectivity_observed: float | None = Field(default=None, ge=0, le=1)
    is_warmup: bool | None = None

    batch_index: int | None = Field(default=None, ge=0)
    batch_size: int | None = Field(default=None, ge=0)
    cumulative_vectors: int | None = Field(default=None, ge=0)
    batch_latency_ms: float | None = Field(default=None, ge=0)
    throughput_vps: float | None = Field(default=None, ge=0)

    ram_mb: float | None = Field(default=None, ge=0)
    ram_after_mb: float | None = Field(default=None, ge=0)
    error: str | None = None
    concurrency_level: int | None = Field(default=None, ge=1)
    search_params: dict[str, Any] | None = None
    dataset_size: int | None = Field(default=None, ge=0)


def _table_from_records(records: list[RawRecord]) -> pa.Table:
    return pa.Table.from_pylist([record.model_dump(mode="json") for record in records])


def _empty_table() -> pa.Table:
    return _table_from_records(
        [
            RawRecord(
                run_id="",
                engine="",
                workload="",
                phase="query",
                status="success",
                timestamp_utc="",
            )
        ]
    ).slice(0, 0)


def write_raw_records(base_dir: str | Path, records: list[RawRecord]) -> None:
    """Write query and ingest micro-data files in PRD03-compatible layout."""

    target_dir = Path(base_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    run_id = records[0].run_id if records else "unknown"
    query_path = target_dir / f"raw_queries_{run_id}.parquet"
    ingest_path = target_dir / f"raw_ingest_{run_id}.parquet"

    query_records = [record for record in records if "query" in record.phase]
    ingest_records = [record for record in records if "ingest" in record.phase]

    query_table = _table_from_records(query_records) if query_records else _empty_table()
    ingest_table = _table_from_records(ingest_records) if ingest_records else _empty_table()
    pq.write_table(query_table, query_path)
    pq.write_table(ingest_table, ingest_path)


def load_raw_records(base_dir: str | Path, **filters: Any) -> list[dict[str, Any]]:
    """Load query and ingest raw records with optional exact-match filters."""

    target_dir = Path(base_dir)
    if not target_dir.exists():
        return []

    tables: list[pa.Table] = []
    for path in sorted(target_dir.glob("raw_*_run-*.parquet")):
        tables.append(pq.read_table(path))
    if not tables:
        return []

    merged = pa.concat_tables(tables, promote_options="default")
    rows = cast(list[dict[str, Any]], merged.to_pylist())

    if not filters:
        return rows

    filtered: list[dict[str, Any]] = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            filtered.append(row)
    return filtered
