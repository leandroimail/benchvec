"""Unit tests for configuration parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchvec.config import BenchmarkConfig, load_config


@pytest.mark.unit
def test_benchmark_config_defaults() -> None:
    config = BenchmarkConfig()

    assert config.dataset.primary == "fast_wikipedia"
    assert config.workloads.concurrent_workers == [1]
    assert config.timeouts.ingest_batch_seconds == 1800


@pytest.mark.unit
def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "benchvec.yaml"
    config_path.write_text(
        """
project_name: benchvec-test
seed: 99
dataset:
  primary: simple_wikipedia
workloads:
  repetitions: 2
  concurrent_workers: [1, 4]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_name == "benchvec-test"
    assert config.seed == 99
    assert config.dataset.primary == "simple_wikipedia"
    assert config.workloads.concurrent_workers == [1, 4]


@pytest.mark.unit
def test_rejects_invalid_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "benchvec.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported configuration format"):
        load_config(config_path)
