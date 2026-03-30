"""Unit tests for Parquet-backed raw measurement storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.raw_store import RawRecord, load_raw_records, write_raw_records


@pytest.mark.unit
def test_write_and_load_raw_records(tmp_path: Path) -> None:
    records = [
        RawRecord(
            run_id="run-1",
            engine="faiss",
            workload="a1",
            phase="ingest",
            status="success",
            timestamp_utc="2026-03-29T12:00:00Z",
            latency_ms=12.3,
            throughput_vps=1234.5,
        ),
        RawRecord(
            run_id="run-1",
            engine="faiss",
            workload="a1",
            phase="query",
            status="success",
            timestamp_utc="2026-03-29T12:00:01Z",
            query_id="query-1",
            latency_ms=8.7,
        ),
    ]

    write_raw_records(tmp_path / "raw", records)
    loaded_records = load_raw_records(tmp_path / "raw", run_id="run-1", engine="faiss")

    assert len(loaded_records) == 2
    assert {record["phase"] for record in loaded_records} == {"ingest", "query"}
