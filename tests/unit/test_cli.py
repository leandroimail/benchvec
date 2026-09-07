"""Unit tests for the BenchVec CLI surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from benchvec import cli
from benchvec.cli import app
from benchvec.datasets import PreparedDatasetManifest, PreparedRecord
from benchvec.metrics import BenchmarkSummary, write_env_snapshot, write_summary

cli_runner = CliRunner()


def make_summary(run_id: str, workload: str) -> BenchmarkSummary:
    return BenchmarkSummary(
        run_id=run_id,
        timestamp="2026-03-29T12:00:00Z",
        engine="faiss",
        engine_version="test",
        dataset_name="cohere_wikipedia",
        dataset_group="rag",
        dataset_size=4,
        dimension=3,
        index_type="flat",
        distance_metric="l2",
        workload=workload,
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


@pytest.mark.unit
def test_report_command_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "faiss" / "a2" / "run-1"
    write_summary(run_dir, make_summary("run-1", "a2"))
    write_env_snapshot(run_dir, {"run_id": "run-1"})

    result = cli_runner.invoke(
        app,
        [
            "report",
            "--input",
            str(tmp_path / "results"),
            "--output",
            str(tmp_path / "final.csv"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "final.csv").exists()
    assert (tmp_path / "results" / "index.parquet").exists()


@pytest.mark.unit
def test_inspect_command_prints_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "faiss" / "a2" / "run-1"
    write_summary(run_dir, make_summary("run-1", "a2"))
    write_env_snapshot(run_dir, {"run_id": "run-1", "host_fingerprint": {"os_name": "Darwin"}})

    result = cli_runner.invoke(
        app,
        [
            "inspect",
            "--input",
            str(tmp_path / "results"),
            "--run-id",
            "run-1",
            "--summary",
        ],
    )

    assert result.exit_code == 0
    assert "run-1" in result.stdout
    assert "mean_latency_ms" in result.stdout


@pytest.mark.unit
def test_run_command_dispatches_selected_workload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchvec.toml"
    config_path.write_text(
        """
project_name = "benchvec-test"

[paths]
benchmark_dir = ".benchmarks"
cache_dir = ".cache"

[dataset]
primary = "cohere_wikipedia"
sample_size = 2

[engines]
enabled = ["faiss"]

[workloads]
k_values = [10]
ingest_batch_sizes = [2]
ingest_rates_per_second = [100]
""".strip(),
        encoding="utf-8",
    )

    records = [
        PreparedRecord(
            id="doc-1",
            text="Document 1",
            embedding=[0.0, 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title="Doc 1",
            document_length_bucket="short",
            categoria="data_engineering",
            tag="engenharia_dados",
            autor="engenharia",
            data_criacao=1704067200,
        ),
        PreparedRecord(
            id="doc-2",
            text="Document 2",
            embedding=[1.0, 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title="Doc 2",
            document_length_bucket="short",
            categoria="science",
            tag="ciencia",
            autor="pesquisa",
            data_criacao=1704153600,
        ),
    ]
    manifest = PreparedDatasetManifest(
        dataset_name="cohere",
        dataset_variant="en",
        sample_size=2,
        seed=42,
    )

    class FakeRunner:
        def __init__(
            self,
            config: object,
            *,
            project: object | None = None,
            clean_engine_storage: bool = True,
        ) -> None:
            del config, project, clean_engine_storage

        def default_query_records(
            self,
            records: list[PreparedRecord],
        ) -> list[PreparedRecord]:
            return records

        def build_filter_scenarios(self, records: list[PreparedRecord]) -> list[object]:
            class Scenario:
                filters = object()

            del records
            return [Scenario()]

        def run_workload_a1(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            batch_size: int,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, batch_size, repetition_index
            return make_summary("run-a1", "a1")

        def run_workload_a2(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            query_records: list[PreparedRecord],
            k: int,
            ground_truth: object = None,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, query_records, k, ground_truth, repetition_index
            return make_summary("run-a2", "a2")

        def run_workload_b1(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            filters: object,
            query_records: list[PreparedRecord],
            k: int,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, filters, query_records, k, repetition_index
            return make_summary("run-b1", "b1")

        def run_workload_b2(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            query_records: list[PreparedRecord],
            batch_size: int,
            ingest_rate_per_second: int,
            k: int,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del (
                engine_name,
                records,
                query_records,
                batch_size,
                ingest_rate_per_second,
                k,
                repetition_index,
            )
            return make_summary("run-b2", "b2")

        def run_workload_b3(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            query_records: list[PreparedRecord],
            k: int,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, query_records, k, repetition_index
            return make_summary("run-b3", "b3")

        def run_workload_b4(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            query_records: list[PreparedRecord],
            k: int,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, query_records, k, repetition_index
            return make_summary("run-b4", "b4")

    monkeypatch.setattr(
        "benchvec.cli.prepare_dataset_from_config",
        lambda dataset_config, cache_dir: (manifest, records),
    )
    monkeypatch.setattr("benchvec.cli.BenchmarkRunner", FakeRunner)

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--workload",
            "a2",
        ],
    )

    assert result.exit_code == 0
    assert "run-a2" in result.stdout


@pytest.mark.unit
def test_download_command_uses_resolved_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchvec.toml"
    config_path.write_text("project_name = \"benchvec-test\"", encoding="utf-8")

    dataset_config = SimpleNamespace(primary="cohere_wikipedia")
    fake_config = SimpleNamespace(
        dataset=dataset_config,
        paths=SimpleNamespace(cache_dir=tmp_path / ".cache"),
    )

    manifest = PreparedDatasetManifest(
        dataset_name="cohere_wikipedia",
        dataset_variant="en",
        sample_size=2,
        seed=42,
    )
    records = [
        PreparedRecord(
            id="doc-1",
            text="Document 1",
            embedding=[0.0, 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title="Doc 1",
            document_length_bucket="short",
            categoria="data_engineering",
            tag="engenharia_dados",
            autor="engenharia",
            data_criacao=1704067200,
        )
    ]

    captured: dict[str, Path] = {}

    def fake_download(
        config: object,
        cache_dir: Path,
    ) -> tuple[PreparedDatasetManifest, list[PreparedRecord]]:
        del config
        captured["cache_dir"] = cache_dir
        return manifest, records

    monkeypatch.setattr("benchvec.cli.load_config", lambda _path: fake_config)
    monkeypatch.setattr("benchvec.cli.download_dataset_from_config", fake_download)

    result = cli_runner.invoke(
        app,
        [
            "download",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["cache_dir"] == tmp_path / ".cache" / "cohere_wikipedia"
    assert "dataset=cohere_wikipedia" in result.stdout
    assert "records=1" in result.stdout


@pytest.mark.unit
def test_run_command_k_sweep_produces_summary_per_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchvec.toml"
    config_path.write_text(
        """
project_name = "benchvec-k-sweep"

[paths]
benchmark_dir = ".benchmarks"
cache_dir = ".cache"

[dataset]
primary = "cohere_wikipedia"
sample_size = 2

[engines]
enabled = ["faiss"]

[workloads]
repetitions = 1
k_values = [1, 2, 4]
ingest_batch_sizes = [2]
ingest_rates_per_second = [100]
""".strip(),
        encoding="utf-8",
    )

    fake_records = [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(index), 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering",
            tag="engenharia_dados",
            autor="engenharia",
            data_criacao=1704067200 + index,
        )
        for index in range(4)
    ]
    monkeypatch.setattr(
        cli, "prepare_dataset_from_config", lambda dataset, cache_dir: (None, fake_records)
    )

    executed_ks: list[int] = []

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def default_query_records(self, records: list[PreparedRecord]) -> list[PreparedRecord]:
            return records

        def build_filter_scenarios(self, records: list[PreparedRecord]) -> list[object]:
            del records
            return []

        def ground_truth_for_queries(
            self,
            records: list[PreparedRecord],
            queries: list[PreparedRecord],
            *,
            k: int,
        ) -> dict[str, list[str]]:
            gt_calls.append(k)
            return {}

        def run_workload_a2(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            *,
            query_records: list[PreparedRecord],
            k: int,
            ground_truth: object = None,
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, query_records, ground_truth, repetition_index
            executed_ks.append(k)
            return make_summary(f"run-a2-k{k}", "a2")

    gt_calls: list[int] = []
    fake_runner = FakeRunner()
    monkeypatch.setattr("benchvec.cli.BenchmarkRunner", lambda *a, **kw: fake_runner)
    result = cli_runner.invoke(cli.run_command, ["--config", str(config_path), "--workload", "a2"])

    assert result.exit_code == 0
    assert executed_ks == [1, 2, 4]
    # Ground truth is computed exactly once (at max k), then sliced per k.
    assert gt_calls == [4]


def test_c1_invocation_cleans_shared_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """After a c1 run the CLI removes .benchmarks/shared/c1 (review finding 1)."""
    config_path = tmp_path / "benchvec.toml"
    config_path.write_text(
        """
project_name = "benchvec-c1-cleanup"

[paths]
benchmark_dir = "benchmarks"
cache_dir = ".cache"

[dataset]
primary = "cohere_wikipedia"
sample_size = 2

[engines]
enabled = ["faiss"]

[workloads]
repetitions = 1
k_values = [2]
concurrent_workers = [2]
ingest_batch_sizes = [2]
ingest_rates_per_second = [100]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmarks" / "shared" / "c1" / "faiss").mkdir(parents=True)
    (tmp_path / "benchmarks" / "shared" / "c1" / "faiss" / "stale.bin").write_bytes(b"x")

    fake_records = [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(index), 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering",
            tag="engenharia_dados",
            autor="engenharia",
            data_criacao=1704067200 + index,
        )
        for index in range(2)
    ]

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def default_query_records(self, records: list[PreparedRecord]) -> list[PreparedRecord]:
            return records

        def build_filter_scenarios(self, records: list[PreparedRecord]) -> list[object]:
            del records
            return []

        def run_workload_a1(self, *args: object, **kwargs: object) -> BenchmarkSummary:
            return make_summary("run-a1", "a1")

        def run_workload_c1(self, *args: object, **kwargs: object) -> list[BenchmarkSummary]:
            return [make_summary("run-c1", "c1")]

    monkeypatch.setattr(cli, "prepare_dataset_from_config", lambda dataset, cache_dir: (None, fake_records))
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *a, **kw: FakeRunner())
    result = cli_runner.invoke(cli.run_command, ["--config", str(config_path), "--workload", "c1"])

    assert result.exit_code == 0
    assert not (tmp_path / "benchmarks" / "shared" / "c1").exists()


@pytest.mark.unit
def test_run_command_turbovec_sweep_grid_uses_bit_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The turbovec sweep grid flows from turbovec_bit_width (task 1.5)."""
    config_path = tmp_path / "benchvec.toml"
    config_path.write_text(
        """
project_name = "benchvec-turbovec-sweep"

[paths]
benchmark_dir = ".benchmarks"
cache_dir = ".cache"

[dataset]
primary = "cohere_wikipedia"
sample_size = 2

[engines]
enabled = ["turbovec"]
distance_metric = "cosine"

[engines.turbovec]
bit_width = 4

[workloads]
repetitions = 1
k_values = [10]

[workloads.param_sweep]
turbovec_bit_width = [2, 3]
""".strip(),
        encoding="utf-8",
    )

    fake_records = [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(index), 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering",
            tag="engenharia_dados",
            autor="engenharia",
            data_criacao=1704067200 + index,
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        cli, "prepare_dataset_from_config", lambda dataset, cache_dir: (None, fake_records)
    )

    sweep_calls: list[tuple[str, dict[str, list[int]]]] = []

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def default_query_records(self, records: list[PreparedRecord]) -> list[PreparedRecord]:
            return records

        def build_filter_scenarios(self, records: list[PreparedRecord]) -> list[object]:
            del records
            return []

        def run_param_sweep(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            *,
            query_records: list[PreparedRecord],
            k: int,
            grid: dict[str, list[int]],
            max_points: int,
        ) -> BenchmarkSummary | None:
            del records, query_records, k, max_points
            sweep_calls.append((engine_name, dict(grid)))
            return make_summary(f"run-sweep-{len(sweep_calls)}", "sweep")

    monkeypatch.setattr(
        "benchvec.cli.BenchmarkRunner", lambda *a, **kw: FakeRunner()
    )
    result = cli_runner.invoke(cli.run_command, ["--config", str(config_path), "--workload", "sweep"])

    assert result.exit_code == 0
    assert sweep_calls == [("turbovec", {"bit_width": [2, 3]})]


@pytest.mark.unit
def test_run_command_turbovec_empty_grid_still_invokes_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without a configured width grid the sweep is invoked with {} (no-op)."""
    config_path = tmp_path / "benchvec.toml"
    config_path.write_text(
        """
project_name = "benchvec-turbovec-nosweep"

[paths]
benchmark_dir = ".benchmarks"
cache_dir = ".cache"

[dataset]
primary = "cohere_wikipedia"
sample_size = 2

[engines]
enabled = ["turbovec"]
distance_metric = "cosine"

[workloads]
repetitions = 1
k_values = [10]
""".strip(),
        encoding="utf-8",
    )

    fake_records = [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(index), 0.0, 0.0],
            source_dataset="cohere_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering",
            tag="engenharia_dados",
            autor="engenharia",
            data_criacao=1704067200 + index,
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        cli, "prepare_dataset_from_config", lambda dataset, cache_dir: (None, fake_records)
    )

    sweep_calls: list[dict[str, list[int]]] = []

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def default_query_records(self, records: list[PreparedRecord]) -> list[PreparedRecord]:
            return records

        def build_filter_scenarios(self, records: list[PreparedRecord]) -> list[object]:
            del records
            return []

        def run_param_sweep(
            self,
            engine_name: str,
            records: list[PreparedRecord],
            *,
            query_records: list[PreparedRecord],
            k: int,
            grid: dict[str, list[int]],
            max_points: int,
        ) -> BenchmarkSummary | None:
            del engine_name, records, query_records, k, max_points
            sweep_calls.append(dict(grid))
            return None

    monkeypatch.setattr(
        "benchvec.cli.BenchmarkRunner", lambda *a, **kw: FakeRunner()
    )
    result = cli_runner.invoke(cli.run_command, ["--config", str(config_path), "--workload", "sweep"])

    assert result.exit_code == 0
    assert sweep_calls == [{}]
