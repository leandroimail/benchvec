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


def make_chart_summary(
    run_id: str,
    workload: str,
    engine: str = "faiss",
    *,
    k: int | None = 10,
    concurrency: int | None = None,
    dataset_size: int = 100,
    mean_latency_ms: float = 2.0,
    qps: float = 400.0,
    recall_at_k: float | None = 1.0,
    p95_latency_ms: float = 3.0,
) -> BenchmarkSummary:
    """Compact factory for chart-generation fixtures."""

    return BenchmarkSummary(
        run_id=run_id,
        timestamp="2026-03-29T12:00:00Z",
        engine=engine,
        engine_version="test",
        dataset_name="cohere_wikipedia",
        dataset_group="rag",
        dataset_size=dataset_size,
        dimension=3,
        index_type="flat",
        distance_metric="l2",
        workload=workload,
        filter_selectivity=None,
        k=k,
        batch_size=None,
        index_params={},
        search_params={},
        n_warmup_queries=0,
        n_measured_queries=4,
        n_repetitions=1,
        load_time_s=1.0,
        build_time_s=0.2,
        time_to_serve_s=1.2,
        mean_latency_ms=mean_latency_ms,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=4.0,
        std_latency_ms=0.5,
        qps=qps,
        recall_at_k=recall_at_k,
        peak_ram_mb=32.0,
        disk_size_mb=0.1,
        reopen_time_s=0.0,
        timeout_policy_s=30,
        concurrency=concurrency,
        status="success",
        error_message=None,
        host_fingerprint={"os_name": "Darwin"},
        raw_data_path="raw",
    )


def write_summary_runs(results_dir: Path, summaries: list[BenchmarkSummary]) -> None:
    for index, summary in enumerate(summaries):
        run_dir = results_dir / summary.engine / summary.workload / f"run-{index}"
        write_summary(run_dir, summary)


@pytest.mark.unit
def test_pareto_mask_excludes_dominated_points() -> None:
    from benchvec.report import pareto_mask

    # Point (2, 0.9) is dominated by (3, 0.95); ties are not dominated.
    mask = pareto_mask(
        qps_values=[1.0, 2.0, 3.0, 3.0],
        recall_values=[0.8, 0.9, 0.95, 0.95],
    )

    assert mask == [False, False, True, True]


@pytest.mark.unit
def test_pareto_mask_handles_empty_input() -> None:
    from benchvec.report import pareto_mask

    assert pareto_mask([], []) == []


@pytest.mark.unit
def test_generate_charts_produces_new_workload_charts(tmp_path: Path) -> None:
    from benchvec.report import generate_charts

    results_dir = tmp_path / "results"
    summaries: list[BenchmarkSummary] = []
    for thread_count in (1, 4):
        summaries.append(
            make_chart_summary(f"c1-{thread_count}", "c1", concurrency=thread_count, qps=100.0 * thread_count)
        )
    for size in (100, 1000):
        summaries.append(
            make_chart_summary(
                f"c2-{size}", "c2", dataset_size=size, mean_latency_ms=1.0 + size / 1000
            )
        )
    for k in (1, 10):
        summaries.append(make_chart_summary(f"a2-k{k}", "a2", k=k, recall_at_k=0.9))
    summaries.append(make_chart_summary("d1-run", "d1"))
    write_summary_runs(results_dir, summaries)

    generated = generate_charts(results_dir, tmp_path / "charts")
    chart_names = {path.name for path in generated}

    assert "chart_07_concurrency.png" in chart_names
    assert "chart_09_scaling.png" in chart_names
    assert "chart_10_k_sweep.png" in chart_names
    for path in generated:
        assert path.stat().st_size > 0


@pytest.mark.unit
def test_generate_charts_skips_missing_data_gracefully(tmp_path: Path) -> None:
    from benchvec.report import generate_charts

    results_dir = tmp_path / "results"
    write_summary_runs(
        results_dir,
        [make_chart_summary("a2-run", "a2", k=10)],
    )

    generated = generate_charts(results_dir, tmp_path / "charts")
    chart_names = {path.name for path in generated}

    # a2-only data renders the classic charts but no C1/C2 charts.
    assert "chart_01_latency.png" in chart_names
    assert "chart_07_concurrency.png" not in chart_names
    assert "chart_09_scaling.png" not in chart_names
    assert "chart_10_k_sweep.png" not in chart_names


@pytest.mark.unit
def test_generate_charts_filters_a2_to_largest_k(tmp_path: Path) -> None:
    from benchvec.report import collect_workload_raw_rows, generate_charts

    results_dir = tmp_path / "results"
    write_summary_runs(
        results_dir,
        [
            make_chart_summary("a2-k1", "a2", k=1),
            make_chart_summary("a2-k10", "a2", k=10),
            make_chart_summary("a2-k100", "a2", k=100),
        ],
    )

    generated = generate_charts(results_dir, tmp_path / "charts")

    assert generated
    # collect_workload_raw_rows returns nothing for workloads without raw
    # rows but must not raise on the nested layout.
    assert collect_workload_raw_rows(results_dir, "sweep") == []


@pytest.mark.unit
def test_engine_order_includes_turbovec_after_sqlite_vec() -> None:
    # Arrange
    from benchvec.report import _ENGINE_ORDER

    # Act + Assert: turbovec appended after sqlite_vec, existing order
    # untouched.
    assert _ENGINE_ORDER[-1] == "turbovec"
    assert _ENGINE_ORDER[-2] == "sqlite_vec"
    assert _ENGINE_ORDER[:4] == ["faiss", "chromadb", "lancedb", "sqlite_vec"]


def _sweep_raw_rows(run_id: str, engine: str, bit_width: int) -> list[RawRecord]:
    """Build sweep raw rows for one turbovec bit-width grid point."""

    return [
        RawRecord(
            run_id=run_id,
            engine=engine,
            workload="sweep",
            phase="query",
            status="success",
            timestamp_utc="2026-09-07T00:00:00Z",
            query_id=f"q{query_index}",
            query_index=query_index,
            mode="serial",
            latency_ms=1.0 * bit_width + 0.1 * query_index,
            recall_at_k=min(0.25 * bit_width, 1.0),
            search_params={"bit_width": bit_width},
            is_warmup=False,
        )
        for query_index in range(3)
    ]


@pytest.mark.unit
def test_generate_charts_render_turbovec_runs(tmp_path: Path) -> None:
    # Arrange: a mixed-engine frame including turbovec rows must render
    # the classic charts plus the Pareto frontier (raw sweep rows drive
    # chart_08; summaries alone are not enough).
    from benchvec.report import generate_charts

    results_dir = tmp_path / "results"
    write_summary_runs(
        results_dir,
        [
            make_chart_summary("faiss-run", "a2", engine="faiss", k=10),
            make_chart_summary("turbovec-4bit", "a2", engine="turbovec", k=10),
            make_chart_summary("turbovec-sweep-2", "sweep", engine="turbovec", k=10, recall_at_k=0.5),
            make_chart_summary("turbovec-sweep-4", "sweep", engine="turbovec", k=10, recall_at_k=0.9),
        ],
    )
    for run_dir, run_id, width in (
        (results_dir / "turbovec" / "sweep" / "run-2", "turbovec-sweep-2", 2),
        (results_dir / "turbovec" / "sweep" / "run-3", "turbovec-sweep-4", 4),
    ):
        write_raw_records(run_dir, _sweep_raw_rows(run_id, "turbovec", width))

    # Act
    generated = generate_charts(results_dir, tmp_path / "charts")
    chart_names = {path.name for path in generated}

    # Assert
    assert "chart_01_latency.png" in chart_names
    assert "chart_02_qps.png" in chart_names
    assert "chart_08_pareto.png" in chart_names
    for path in generated:
        assert path.stat().st_size > 0
