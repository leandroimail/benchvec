"""Unit tests for summary consolidation and run inspection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchvec.metrics import BenchmarkSummary, write_env_snapshot, write_summary
from benchvec.raw_store import RawRecord, write_raw_records
from benchvec.report import (
    collect_summary_paths,
    consolidate_summaries,
    inspect_run,
    write_consolidated_report,
    write_results_index,
)


@pytest.mark.unit
def test_report_helpers_collect_and_consolidate(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "faiss" / "a2" / "run-1"
    raw_dir = run_dir / "raw"
    summary = BenchmarkSummary(
        run_id="run-1",
        timestamp="2026-03-29T12:00:00Z",
        engine="faiss",
        engine_version="test",
        dataset_name="cohere_wikipedia",
        dataset_group="rag",
        dataset_size=4,
        dimension=3,
        index_type="flat",
        distance_metric="l2",
        workload="a2",
        filter_selectivity=None,
        k=10,
        batch_size=None,
        index_params={},
        search_params={},
        n_warmup_queries=0,
        n_measured_queries=4,
        n_repetitions=1,
        load_time_s=1.0,
        build_time_s=0.2,
        time_to_serve_s=1.2,
        mean_latency_ms=2.0,
        p95_latency_ms=3.0,
        p99_latency_ms=4.0,
        std_latency_ms=0.5,
        qps=400.0,
        recall_at_k=1.0,
        peak_ram_mb=32.0,
        disk_size_mb=0.1,
        reopen_time_s=0.0,
        timeout_policy_s=30,
        status="success",
        error_message=None,
        host_fingerprint={"os_name": "Darwin"},
        raw_data_path="raw",
    )
    write_summary(run_dir, summary)
    write_env_snapshot(run_dir, {"run_id": "run-1", "host_fingerprint": {"os_name": "Darwin"}})
    write_raw_records(
        raw_dir,
        [
            RawRecord(
                run_id="run-1",
                engine="faiss",
                workload="a2",
                phase="query",
                status="success",
                timestamp_utc="2026-03-29T12:00:00Z",
                latency_ms=1.0,
            )
        ],
    )

    summaries = consolidate_summaries(tmp_path / "results")

    assert len(collect_summary_paths(tmp_path / "results")) == 1
    assert summaries[0].run_id == "run-1"


@pytest.mark.unit
def test_write_consolidated_report_and_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "faiss" / "a2" / "run-1"
    summary = BenchmarkSummary(
        run_id="run-1",
        timestamp="2026-03-29T12:00:00Z",
        engine="faiss",
        engine_version="test",
        dataset_name="cohere_wikipedia",
        dataset_group="rag",
        dataset_size=4,
        dimension=3,
        index_type="flat",
        distance_metric="l2",
        workload="a2",
        filter_selectivity=None,
        k=10,
        batch_size=None,
        index_params={},
        search_params={},
        n_warmup_queries=0,
        n_measured_queries=4,
        n_repetitions=1,
        load_time_s=1.0,
        build_time_s=0.2,
        time_to_serve_s=1.2,
        mean_latency_ms=2.0,
        p95_latency_ms=3.0,
        p99_latency_ms=4.0,
        std_latency_ms=0.5,
        qps=400.0,
        recall_at_k=1.0,
        peak_ram_mb=32.0,
        disk_size_mb=0.1,
        reopen_time_s=0.0,
        timeout_policy_s=30,
        status="success",
        error_message=None,
        host_fingerprint={"os_name": "Darwin"},
        raw_data_path="raw",
    )
    write_summary(run_dir, summary)
    write_env_snapshot(run_dir, {"run_id": "run-1"})

    report_path = write_consolidated_report(tmp_path / "results", tmp_path / "report.csv")
    index_path = write_results_index(tmp_path / "results", tmp_path / "index.parquet")

    report_frame = pd.read_csv(report_path)
    index_frame = pd.read_parquet(index_path)

    assert list(report_frame["run_id"]) == ["run-1"]
    assert list(index_frame["run_id"]) == ["run-1"]


@pytest.mark.unit
def test_inspect_run_loads_summary_and_raw_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "faiss" / "a2" / "run-1"
    summary = BenchmarkSummary(
        run_id="run-1",
        timestamp="2026-03-29T12:00:00Z",
        engine="faiss",
        engine_version="test",
        dataset_name="cohere_wikipedia",
        dataset_group="rag",
        dataset_size=4,
        dimension=3,
        index_type="flat",
        distance_metric="l2",
        workload="a2",
        filter_selectivity=None,
        k=10,
        batch_size=None,
        index_params={},
        search_params={},
        n_warmup_queries=0,
        n_measured_queries=4,
        n_repetitions=1,
        load_time_s=1.0,
        build_time_s=0.2,
        time_to_serve_s=1.2,
        mean_latency_ms=2.0,
        p95_latency_ms=3.0,
        p99_latency_ms=4.0,
        std_latency_ms=0.5,
        qps=400.0,
        recall_at_k=1.0,
        peak_ram_mb=32.0,
        disk_size_mb=0.1,
        reopen_time_s=0.0,
        timeout_policy_s=30,
        status="success",
        error_message=None,
        host_fingerprint={"os_name": "Darwin"},
        raw_data_path="raw_queries_run-1.parquet",
    )
    write_summary(run_dir, summary)
    write_env_snapshot(run_dir, {"run_id": "run-1", "host_fingerprint": {"os_name": "Darwin"}})
    # write raw records directly into run_dir (PRD03 layout)
    write_raw_records(
        run_dir,
        [
            RawRecord(
                run_id="run-1",
                engine="faiss",
                workload="a2",
                phase="query",
                status="success",
                timestamp_utc="2026-03-29T12:00:00Z",
                latency_ms=1.0,
            )
        ],
    )

    payload = inspect_run(tmp_path / "results", "run-1")

    assert payload["summary"].run_id == "run-1"
    assert len(payload["raw_records"]) == 1
