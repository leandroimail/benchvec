"""Unit tests for benchmark workload orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pytest

from benchvec.config import BenchmarkConfig
from benchvec.datasets import PreparedRecord, stable_hash
from benchvec.engines.base import (
    BenchmarkEngine,
    DistanceMetric,
    EngineRecord,
    EngineRuntime,
    FilterClause,
    MetadataFilter,
    SearchHit,
)
from benchvec.engines.turbovec_engine import TurboVecEngine, turbovec_available
from benchvec.metrics import load_summary
from benchvec.runner import BenchmarkRunner, record_to_engine_record, record_to_metadata


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

    def delete(self, ids: Sequence[str]) -> int:
        id_set = set(ids)
        before = len(self._records)
        self._records = [record for record in self._records if record.id not in id_set]
        return before - len(self._records)

    def upsert(self, records: Sequence[EngineRecord]) -> None:
        materialized = list(records)
        self.delete([record.id for record in materialized])
        self._records.extend(materialized)

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


@pytest.mark.unit
def test_runner_cleans_engine_storage_by_default(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()

    summary = runner.run_workload_a2("faiss", records, k=2)
    run_dir = tmp_path / "benchmarks" / "results" / "faiss" / "a2" / summary.run_id

    assert summary.disk_size_mb > 0
    assert not (run_dir / "engine").exists()
    assert (run_dir / "summary.parquet").exists()


@pytest.mark.unit
def test_runner_keeps_engine_storage_when_configured(tmp_path: Path) -> None:
    config = BenchmarkConfig()
    config.paths.benchmark_dir = tmp_path / "benchmarks"
    config.workloads.warmup_queries = 2
    config.workloads.repetitions = 1
    config.dataset.ground_truth_query_count = 4
    runner = BenchmarkRunner(
        config,
        engine_factory=lambda engine_name, storage_path: FakeEngine(storage_path),
        clean_engine_storage=False,
    )
    records = make_prepared_records()

    summary = runner.run_workload_a2("faiss", records, k=2)
    run_dir = tmp_path / "benchmarks" / "results" / "faiss" / "a2" / summary.run_id

    assert (run_dir / "engine").exists()
    assert summary.disk_size_mb > 0


@pytest.mark.unit
def test_a2_precomputed_ground_truth_slice_matches_direct_compute(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()
    queries = runner.default_query_records(records, query_count=3)
    k_max = 4
    full_truth = runner.ground_truth_for_queries(records, queries, k=k_max)
    sliced = {qid: ids[:2] for qid, ids in full_truth.items()}

    summary_full = runner.run_workload_a2(
        "fake", records, query_records=queries, k=4, ground_truth=full_truth
    )
    summary_slice = runner.run_workload_a2(
        "fake", records, query_records=queries, k=2, ground_truth=sliced
    )
    summary_direct = runner.run_workload_a2("fake", records, query_records=queries, k=2)

    # The k-slice of the rank-ordered map is the exact k-NN answer.
    assert summary_slice.recall_at_k == pytest.approx(summary_direct.recall_at_k)
    assert summary_full.k == 4 and summary_slice.k == 2
    assert summary_direct.recall_at_k == pytest.approx(1.0)


def _config_for_runner(tmp_path: Path) -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.paths.benchmark_dir = tmp_path / "benchmarks"
    config.workloads.warmup_queries = 2
    config.workloads.repetitions = 1
    config.dataset.ground_truth_query_count = 4
    return config


class SlowFakeEngine(FakeEngine):
    """FakeEngine whose search sleeps, releasing the GIL like real engines."""

    def __init__(self, storage_path: Path, *, search_seconds: float = 0.02) -> None:
        super().__init__(storage_path)
        self.search_seconds = search_seconds

    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        time.sleep(self.search_seconds)
        return self._search_records(self._records, query, k)


class BrokenConcurrentEngine(FakeEngine):
    """Fails under concurrency once, succeeds under the locked retry."""

    def __init__(self, storage_path: Path) -> None:
        super().__init__(storage_path)
        self.active = 0
        self.failed_once = False

    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        self.active += 1
        try:
            # Brief window so truly overlapping calls trip the failure.
            time.sleep(0.005)
            if self.active > 1 and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("not thread-safe")
            return self._search_records(self._records, query, k)
        finally:
            self.active -= 1


def make_prepared_records_concurrent(count: int) -> list[PreparedRecord]:
    base = make_prepared_records()
    return [base[index % len(base)] for index in range(count)]


@pytest.mark.unit
def test_c1_returns_one_summary_per_thread_count(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()

    summaries = runner.run_workload_c1(
        "fake", records, k=2, thread_counts=[1, 2, 4]
    )

    assert [summary.concurrency for summary in summaries] == [1, 2, 4]
    assert all(summary.workload == "c1" for summary in summaries)
    assert all(summary.recall_at_k == pytest.approx(1.0) for summary in summaries)


@pytest.mark.unit
def test_c1_qps_scales_with_threads_for_gil_releasing_engine(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        _config_for_runner(tmp_path),
        engine_factory=lambda engine_name, storage_path: SlowFakeEngine(storage_path),
    )
    records = make_prepared_records_concurrent(8)

    summaries = runner.run_workload_c1("slow", records, k=1, thread_counts=[1, 4])
    qps_one, qps_four = summaries[0].qps, summaries[1].qps

    # c=1 runs 8 amplified queries × 20ms ≈ 0.16s wall; c=4 runs 32 queries
    # across 4 workers ≈ 0.16s wall with 4× the throughput (sleep releases GIL).
    assert qps_four > qps_one * 2


@pytest.mark.unit
def test_c1_marks_partial_when_engine_needs_locked_retry(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        _config_for_runner(tmp_path),
        engine_factory=lambda engine_name, storage_path: BrokenConcurrentEngine(storage_path),
    )
    records = make_prepared_records_concurrent(8)

    summaries = runner.run_workload_c1("broken", records, k=1, thread_counts=[4])

    assert summaries[0].status == "partial"
    assert summaries[0].error_message is not None
    # The locked retry rescued the queries.
    assert summaries[0].n_measured_queries > 0


class ParamRecordingEngine(FakeEngine):
    """Records set_query_params calls and counts ingest batches."""

    def __init__(self, storage_path: Path) -> None:
        super().__init__(storage_path)
        self.param_history: list[dict[str, Any]] = []
        self.insert_calls = 0

    def set_query_params(self, params: Mapping[str, Any]) -> None:
        self.param_history.append(dict(params))

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        self.insert_calls += 1
        super().insert(batch)


@pytest.mark.unit
def test_param_sweep_runs_every_grid_point_without_reingest(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()
    created_engines: list[ParamRecordingEngine] = []

    def factory(engine_name: str, storage_path: Path) -> ParamRecordingEngine:
        engine = ParamRecordingEngine(storage_path)
        created_engines.append(engine)
        return engine

    runner.engine_factory = factory  # type: ignore[method-assign]
    summary = runner.run_param_sweep(
        "fake",
        records,
        k=2,
        grid={"nprobe": [1, 2, 4]},
        max_points=20_000,
    )

    assert summary is not None and summary.workload == "sweep"
    assert created_engines[0].param_history == [{"nprobe": 1}, {"nprobe": 2}, {"nprobe": 4}]
    # Own ingest once; points apply live with no re-ingest between them.
    assert created_engines[0].insert_calls == 1


@pytest.mark.unit
def test_param_sweep_caps_points_at_budget(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records()
    created_engines: list[ParamRecordingEngine] = []

    def factory(engine_name: str, storage_path: Path) -> ParamRecordingEngine:
        engine = ParamRecordingEngine(storage_path)
        created_engines.append(engine)
        return engine

    runner.engine_factory = factory  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        summary = runner.run_param_sweep(
            "fake",
            records,
            k=2,
            grid={"nprobe": [1, 2, 4, 8]},
            max_points=8,  # 4 points × 4 queries = 16 > 8 → keep 2 points
        )

    assert summary is not None
    # Even subsampling (step 2 over [1,2,4,8]) keeps the grid's ends so the
    # frontier's high-cost region is not amputated.
    assert created_engines[0].param_history == [{"nprobe": 1}, {"nprobe": 4}]
    assert "capped from 4 to 2 points" in caplog.text


@pytest.mark.unit
def test_param_sweep_empty_grid_returns_none(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)

    summary = runner.run_param_sweep(
        "fake", make_prepared_records(), k=2, grid={}, max_points=100
    )

    assert summary is None


@pytest.mark.unit
def test_scaling_ladder_nested_prefixes_and_summaries(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_prepared_records_concurrent(16)

    summaries = runner.run_scaling_ladder(
        "fake",
        records,
        k=2,
        sample_sizes=[4, 8, 4, 999, 0],
        batch_size=2,
        max_c2_queries=20_000,
    )

    # Sizes deduplicated, sorted, clamped to 16, and 0/999 dropped.
    assert [summary.dataset_size for summary in summaries] == [4, 8]
    assert all(summary.workload == "c2" for summary in summaries)
    # Nested-prefix property is enforced by construction; verify determinism
    # by rebuilding the ordering the same way the runner does.
    ordering = sorted(
        records,
        key=lambda record: stable_hash(f"{runner.config.seed}:{record.id}:scale"),
    )
    assert {record.id for record in ordering[:4]} <= {record.id for record in ordering[:8]}


@pytest.mark.unit
def test_scaling_ladder_skipped_when_sizes_empty(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)

    summaries = runner.run_scaling_ladder(
        "fake",
        make_prepared_records(),
        k=2,
        sample_sizes=[],
        batch_size=2,
        max_c2_queries=100,
    )

    assert summaries == []


def make_tie_free_records(count: int) -> list[PreparedRecord]:
    """Records whose pairwise distances are all distinct (no ranking ties)."""

    return [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(2 ** (index + 1)), 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering" if index < 4 else "science",
            tag="engenharia_dados" if index < 4 else "ciencia",
            autor="engenharia" if index % 2 == 0 else "pesquisa",
            data_criacao=1704067200 + (index % 5) * 86400,
            token_count=32 + index,
        )
        for index in range(count)
    ]


@pytest.mark.unit
def test_d1_lifecycle_measures_delete_upsert_and_recall(tmp_path: Path) -> None:
    runner = build_runner(tmp_path)
    records = make_tie_free_records(8)

    summary = runner.run_workload_d1(
        "fake",
        records,
        k=2,
        delete_fraction=0.25,
        upsert_fraction=0.25,
    )

    # Deterministic delete set: first 25% of the stable-hash ordering.
    ordering = sorted(
        records,
        key=lambda record: stable_hash(f"{runner.config.seed}:{record.id}:scale"),
    )
    expected_deleted = {record.id for record in ordering[:2]}

    assert summary.workload == "d1"
    assert summary.delete_time_s is not None and summary.delete_time_s >= 0.0
    assert summary.upsert_time_s is not None and summary.upsert_time_s >= 0.0
    assert summary.post_delete_recall is not None
    # The upsert set includes live overwrites (ranks 10-15%) plus resurrections.
    assert summary.recall_at_k is not None and summary.recall_at_k >= 0.0
    assert expected_deleted  # sanity: delete set non-empty
    # The exact fake must score perfectly against the corrected reference.
    assert summary.post_delete_recall == pytest.approx(1.0)
    assert summary.recall_at_k == pytest.approx(1.0)


@pytest.mark.unit
def test_d1_delete_set_is_deterministic_across_runs(tmp_path: Path) -> None:
    delete_sets = []
    for _ in range(2):
        runner = build_runner(tmp_path / f"run-{_}")
        records = make_prepared_records_concurrent(16)
        ordering = sorted(
            records,
            key=lambda record: stable_hash(f"{runner.config.seed}:{record.id}:scale"),
        )
        delete_sets.append([record.id for record in ordering[:2]])

    assert delete_sets[0] == delete_sets[1]


@pytest.mark.unit
def test_c1_keeps_shared_storage_until_invocation_ends(tmp_path: Path) -> None:
    """C1 shared storage persists during the run (live clients) and is
    removed by the CLI after the invocation (see test_cli.py)."""
    runner = build_runner(tmp_path)

    runner.run_workload_c1("fake", make_prepared_records(), k=2, thread_counts=[1, 2])

    shared_dir = tmp_path / "benchmarks" / "shared" / "c1" / "fake"
    assert shared_dir.exists()




# ---------------------------------------------------------------------------
# TurboVec runner wiring (task 1.5)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not turbovec_available(), reason="turbovec wheel is not available on this host"
)
@pytest.mark.unit
def test_build_engine_wires_turbovec_config(tmp_path: Path) -> None:
    # Arrange
    runner = build_runner(tmp_path)
    runner.config.engines.distance_metric = "cosine"
    runner.config.engines.turbovec.bit_width = 3
    runner.config.engines.turbovec.calibrate = True

    # Act
    engine = runner._build_engine("turbovec", tmp_path / "tv")

    # Assert
    assert isinstance(engine, TurboVecEngine)
    assert engine.distance_metric == "cosine"
    assert engine.bit_width == 3
    assert engine.calibrate_enabled is True
    assert engine.index_type == "turboquant-3bit"


@pytest.mark.skipif(
    not turbovec_available(), reason="turbovec wheel is not available on this host"
)
@pytest.mark.unit
def test_apply_sweep_params_rebuilds_turbovec_bit_width(tmp_path: Path) -> None:
    # Arrange: dim-16 records (turbovec requires dim % 8 == 0); the sweep
    # applies points live and never re-calls build_index.
    runner = build_runner(tmp_path)
    rng = np.random.default_rng(13)
    records = [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=rng.normal(size=16).tolist(),
            source_dataset="fast_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="science",
            tag="ciencia",
            autor="pesquisa",
            data_criacao=1704067200,
        )
        for index in range(6)
    ]
    engine = TurboVecEngine(
        tmp_path / "tve", distance_metric="cosine", bit_width=4, calibrate=True
    )
    engine.insert([record_to_engine_record(r) for r in records])
    engine.build_index()

    # Act: one sweep point — rebuild to 2-bit then re-ingest.
    runner._apply_sweep_params(engine, {"bit_width": 2}, records)

    # Assert: runtime + params reflect the new width, calibration survived
    # the swap (sample retained), and the re-ingested index serves queries.
    assert engine.bit_width == 2
    assert engine.index_type == "turboquant-2bit"
    assert engine._index is not None
    assert engine._index.calibration_state == "calibrated"
    hits = engine.search(records[0].embedding, 3, "cosine")
    assert hits and hits[0].id == "doc-0"
    assert engine.last_search_params["bit_width"] == 2
    assert engine.last_search_params["calibration_state"] == "calibrated"


@pytest.mark.unit
def test_filter_name_is_capability_based() -> None:
    # Arrange: application-layer filterers (faiss, sqlite_vec, turbovec)
    # must be labeled pre-filter regardless of engine name.
    filters = MetadataFilter(
        clauses=(FilterClause(field="tag", operator="eq", value="ciencia"),)
    )

    class NativeEngine:
        name = "some-native-engine"
        native_filtering = True

    class ApplicationEngine:
        name = "some-app-engine"
        native_filtering = False

    # Act + Assert
    assert BenchmarkRunner._filter_name(filters, NativeEngine()) == "native:1-clauses"
    assert (
        BenchmarkRunner._filter_name(filters, ApplicationEngine())
        == "pre-filter:application-layer"
    )
