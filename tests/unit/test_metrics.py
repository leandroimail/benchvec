"""Unit tests for benchmark metric helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.metrics import (
    BenchmarkSummary,
    calculate_recall_at_k,
    load_summary,
    summarize_latencies,
    write_summary,
)


@pytest.mark.unit
def test_summarize_latencies_computes_percentiles() -> None:
    stats = summarize_latencies([10.0, 20.0, 30.0, 40.0])

    assert stats.mean_ms == 25.0
    assert stats.p95_ms >= 38.0
    assert stats.p99_ms >= stats.p95_ms
    assert stats.std_ms > 0


@pytest.mark.unit
def test_calculate_recall_at_k_uses_overlap() -> None:
    recall = calculate_recall_at_k(["a", "x", "b"], ["a", "b", "c"], k=3)

    assert recall == pytest.approx(2 / 3)


@pytest.mark.unit
def test_write_and_load_summary_roundtrip(tmp_path: Path) -> None:
    summary = BenchmarkSummary(
        run_id="run-123",
        timestamp="2026-03-29T12:00:00Z",
        engine="faiss",
        engine_version="1.0.0",
        dataset_name="cohere_wikipedia",
        dataset_group="rag",
        dataset_size=10,
        dimension=4,
        index_type="flat",
        distance_metric="l2",
        workload="a2",
        filter_selectivity=None,
        k=10,
        batch_size=None,
        index_params={},
        search_params={},
        n_warmup_queries=0,
        n_measured_queries=10,
        n_repetitions=1,
        load_time_s=1.0,
        build_time_s=0.5,
        time_to_serve_s=1.5,
        mean_latency_ms=2.0,
        p95_latency_ms=3.0,
        p99_latency_ms=4.0,
        std_latency_ms=0.5,
        qps=500.0,
        recall_at_k=1.0,
        peak_ram_mb=64.0,
        disk_size_mb=1.5,
        reopen_time_s=0.1,
        timeout_policy_s=30,
        status="success",
        error_message=None,
        host_fingerprint={"os_name": "Darwin"},
        raw_data_path="raw",
    )

    target_path = write_summary(tmp_path, summary)
    loaded = load_summary(target_path)

    assert loaded.run_id == summary.run_id
    assert loaded.mean_latency_ms == summary.mean_latency_ms
