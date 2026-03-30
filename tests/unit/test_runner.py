"""Unit tests for benchmark workload orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from benchvec.config import BenchmarkConfig
from benchvec.datasets import PreparedRecord
from benchvec.engines.base import (
    BenchmarkEngine,
    DistanceMetric,
    EngineRecord,
    EngineRuntime,
    MetadataFilter,
    SearchHit,
)
from benchvec.metrics import load_summary
from benchvec.runner import BenchmarkRunner, record_to_metadata


class FakeEngine(BenchmarkEngine):
    """In-memory engine used to validate runner behavior deterministically."""

    def __init__(self, storage_path: Path) -> None:
        self.name = "fake"
        self.distance_metric = "l2"
        self.index_type = "flat"
        self.native_filtering = True
        self.storage_path = storage_path
        self._records: list[EngineRecord] = []
        self._persisted_records: list[EngineRecord] = []

    def connect(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self._records = []
        self._persisted_records = []
        if self.storage_path.exists():
            for item in self.storage_path.iterdir():
                if item.is_file():
                    item.unlink()

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        self._records.extend(batch)

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
        return self._search_records(self._records, query, k)

    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        filtered_records = [record for record in self._records if filters.matches(record.metadata)]
        return self._search_records(filtered_records, query, k)

    def persist(self) -> None:
        self._persisted_records = list(self._records)
        (self.storage_path / "state.txt").write_text(str(len(self._records)), encoding="utf-8")

    def reopen(self) -> None:
        self._records = list(self._persisted_records)

    def disk_size(self) -> int:
        state_path = self.storage_path / "state.txt"
        return state_path.stat().st_size if state_path.exists() else 0

    def describe_runtime(self) -> EngineRuntime:
        return EngineRuntime(
            engine=self.name,
            package_version="test",
            distance_metric="l2",
            index_type="flat",
            native_filtering=True,
            storage_path=str(self.storage_path),
        )

    @staticmethod
    def _search_records(records: list[EngineRecord], query: list[float], k: int) -> list[SearchHit]:
        ranked = sorted(
            records,
            key=lambda record: sum(
                (left - right) ** 2 for left, right in zip(record.embedding, query)
            ),
        )
        return [
            SearchHit(
                id=record.id,
                score=float(
                    sum((left - right) ** 2 for left, right in zip(record.embedding, query))
                ),
                document=record.document,
                metadata=dict(record.metadata),
            )
            for record in ranked[:k]
        ]


def make_prepared_records() -> list[PreparedRecord]:
    return [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(index), float(index % 2), 0.0],
            source_dataset="cohere_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering" if index < 4 else "science",
            tag="engenharia_dados" if index < 4 else "ciencia",
            autor="engenharia" if index % 2 == 0 else "pesquisa",
            data_criacao=1704067200 + (index % 5) * 86400,
            token_count=32 + index,
        )
        for index in range(8)
    ]


def build_runner(tmp_path: Path) -> BenchmarkRunner:
    config = BenchmarkConfig()
    config.paths.benchmark_dir = tmp_path / "benchmarks"
    config.workloads.warmup_queries = 2
    config.workloads.repetitions = 1
    config.dataset.ground_truth_query_count = 4
    return BenchmarkRunner(
        config,
        engine_factory=lambda engine_name, storage_path: FakeEngine(storage_path),
    )


@pytest.mark.unit
def test_build_filter_scenarios_returns_targeted_candidates(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    scenarios = runner.build_filter_scenarios(make_prepared_records())

    assert len(scenarios) == 4
    assert all(0.0 <= scenario.observed_selectivity <= 1.0 for scenario in scenarios)


@pytest.mark.unit
def test_run_workload_a2_persists_summary(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()

    summary = runner.run_workload_a2("faiss", records, k=2)
    summary_path = (
        tmp_path
        / "benchmarks"
        / "results"
        / "faiss"
        / "a2"
        / summary.run_id
        / "summary.parquet"
    )
    loaded = load_summary(summary_path)

    assert summary.workload == "a2"
    assert summary.recall_at_k == pytest.approx(1.0)
    assert loaded.run_id == summary.run_id


@pytest.mark.unit
def test_run_workload_b1_reports_filter_selectivity(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()
    scenario = runner.build_filter_scenarios(records)[0]

    summary = runner.run_workload_b1("faiss", records, filters=scenario.filters, k=2)

    assert summary.workload == "b1"
    assert summary.filter_selectivity is not None
    assert 0.0 <= summary.filter_selectivity <= 1.0


@pytest.mark.unit
def test_run_workload_b3_reopens_engine(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()

    summary = runner.run_workload_b3("faiss", records, k=2)

    assert summary.workload == "b3"
    assert summary.reopen_time_s >= 0.0
    assert summary.recall_at_k == pytest.approx(1.0)


@pytest.mark.unit
def test_record_to_metadata_exposes_expected_filter_fields() -> None:
    metadata = record_to_metadata(make_prepared_records()[0])

    assert metadata["source_dataset"] == "cohere_wikipedia"
    assert metadata["categoria"] == "data_engineering"
    assert metadata["token_count"] == 32


# ---------------------------------------------------------------------------
# BDD threshold tests: timeout behaviour
# ---------------------------------------------------------------------------


class SlowEngine(FakeEngine):
    """Engine whose ingest hangs long enough to trigger a zero-second timeout."""

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        import time

        time.sleep(0.5)
        super().insert(batch)


@pytest.mark.unit
def test_workload_a1_timeout_produces_timed_out_status(tmp_path: Path) -> None:
    """A1 with a zero-second ingest timeout must record status='timed_out'."""
    config = BenchmarkConfig()
    config.paths.benchmark_dir = tmp_path / "benchmarks"
    config.workloads.warmup_queries = 0
    config.workloads.repetitions = 1
    config.dataset.ground_truth_query_count = 2
    config.timeouts.ingest_batch_seconds = 0  # force immediate timeout

    runner = BenchmarkRunner(
        config,
        engine_factory=lambda name, path: SlowEngine(path),
    )
    records = make_prepared_records()
    summary = runner.run_workload_a1("faiss", records, batch_size=8)

    assert summary.status == "timed_out"
    assert summary.workload == "a1"


@pytest.mark.unit
def test_workload_a2_persists_log_file(tmp_path: Path) -> None:
    """Each workload run must produce a benchmark.log file in the run directory."""
    runner = build_runner(tmp_path)
    records = make_prepared_records()

    summary = runner.run_workload_a2("faiss", records, k=2)

    log_path = (
        tmp_path
        / "benchmarks"
        / "results"
        / "faiss"
        / "a2"
        / summary.run_id
        / "benchmark.log"
    )
    assert log_path.exists(), "benchmark.log must be created in run directory"
    log_text = log_path.read_text(encoding="utf-8")
    assert summary.run_id in log_text
