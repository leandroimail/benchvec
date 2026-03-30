"""Unit tests for the BenchVec CLI surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

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
        def __init__(self, config: object, *, project: object | None = None) -> None:
            del config, project

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
            repetition_index: int = 0,
        ) -> BenchmarkSummary:
            del engine_name, records, query_records, k, repetition_index
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
