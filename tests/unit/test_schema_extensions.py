"""Unit tests for the schema extensions added by the expansion plan."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchvec.config import BenchmarkConfig, DatasetConfig, TurboVecConfig, load_config
from benchvec.metrics import BenchmarkSummary
from benchvec.raw_store import RawRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_legacy_toml_configs_still_load() -> None:
    # Arrange + Act: the two committed example configs predate the new schema.
    first = load_config(REPO_ROOT / "configs" / "first-benchmark.toml")
    fast = load_config(REPO_ROOT / "configs" / "fast-benchmark.toml")

    # Assert: defaults fill every new section without user changes.
    assert first.dataset.primary == "cohere_wikipedia"
    assert fast.dataset.primary == "fast_wikipedia"
    assert first.engines.lancedb.index_type == "ivfpq"
    assert first.engines.chroma.search_ef == 64
    assert first.workloads.param_sweep.max_points == 20_000
    assert fast.workloads.sample_sizes == []


@pytest.mark.unit
def test_lancedb_config_defaults_target_indexed_mode() -> None:
    # Arrange + Act
    config = BenchmarkConfig()

    # Assert: the fairness fix defaults engage at the 3k fast profile size.
    assert config.engines.lancedb.index_type == "ivfpq"
    assert config.engines.lancedb.num_partitions == 8
    assert config.engines.lancedb.num_sub_vectors == 16
    assert config.engines.lancedb.nprobes == 8
    assert config.engines.lancedb.refine_factor == 1


@pytest.mark.unit
def test_chroma_config_defaults() -> None:
    # Arrange + Act
    config = BenchmarkConfig()

    # Assert: single Pareto point by default (grid off).
    assert config.engines.chroma.search_ef == 64
    assert config.engines.chroma.ef_search_grid_enabled is False
    assert config.engines.chroma.ef_search_grid == []


@pytest.mark.unit
def test_faiss_nprobe_default_is_zero() -> None:
    # Arrange + Act
    config = BenchmarkConfig()

    # Assert: 0 means "library default" (only meaningful for ivfflat).
    assert config.engines.faiss_nprobe == 0


@pytest.mark.unit
def test_param_sweep_grids_parse_from_toml(tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        """
[workloads.param_sweep]
faiss_nprobe = [1, 2, 4]
lancedb_nprobes = [1, 8]
lancedb_refine_factor = [1, 4]
max_points = 500
""".strip(),
        encoding="utf-8",
    )

    # Act
    config = load_config(config_path)

    # Assert
    sweep = config.workloads.param_sweep
    assert sweep.faiss_nprobe == [1, 2, 4]
    assert sweep.lancedb_nprobes == [1, 8]
    assert sweep.lancedb_refine_factor == [1, 4]
    assert sweep.max_points == 500


@pytest.mark.unit
def test_sample_sizes_and_c2_budget_parse_from_toml(tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "scale.toml"
    config_path.write_text(
        """
[workloads]
sample_sizes = [500, 1000, 2000]
max_c2_queries = 8000
""".strip(),
        encoding="utf-8",
    )

    # Act
    config = load_config(config_path)

    # Assert
    assert config.workloads.sample_sizes == [500, 1000, 2000]
    assert config.workloads.max_c2_queries == 8000


@pytest.mark.unit
def test_sqlite_vec_engine_name_accepted(tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        """
[engines]
enabled = ["sqlite_vec"]
""".strip(),
        encoding="utf-8",
    )

    # Act
    config = load_config(config_path)

    # Assert
    assert config.engines.enabled == ["sqlite_vec"]


@pytest.mark.unit
def test_unknown_engine_name_still_rejected(tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        """
[engines]
enabled = ["postgres"]
""".strip(),
        encoding="utf-8",
    )

    # Act + Assert
    with pytest.raises(ValidationError):
        load_config(config_path)


@pytest.mark.unit
def test_unknown_top_level_key_still_rejected(tmp_path: Path) -> None:
    # Arrange: extra="forbid" must keep rejecting unknown keys.
    config_path = tmp_path / "bad.toml"
    config_path.write_text("unknown_key = true", encoding="utf-8")

    # Act + Assert
    with pytest.raises(ValidationError):
        load_config(config_path)


@pytest.mark.unit
def test_duplicate_dataset_keys_declared_once() -> None:
    # Arrange + Act: source-level check — a duplicated field annotation in
    # the class body is a silent last-wins no-op at runtime, so inspect it.
    source = inspect.getsource(DatasetConfig)

    # Assert: each fast-wikipedia key is declared exactly once.
    assert source.count("fast_wikipedia_dataset_name:") == 1
    assert source.count("fast_wikipedia_split:") == 1


def _minimal_summary_kwargs() -> dict[str, object]:
    """Return the minimum kwargs BenchmarkSummary requires."""
    return {
        "run_id": "run-test",
        "timestamp": "2026-09-06T00:00:00Z",
        "engine": "faiss",
        "engine_version": "1.0",
        "dataset_name": "fast_wikipedia",
        "dataset_group": "rag",
        "dataset_size": 10,
        "dimension": 384,
        "index_type": "flat",
        "distance_metric": "l2",
        "workload": "a2",
        "n_warmup_queries": 0,
        "n_measured_queries": 1,
        "n_repetitions": 1,
        "load_time_s": 0.0,
        "build_time_s": 0.0,
        "time_to_serve_s": 0.0,
        "mean_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "std_latency_ms": 0.0,
        "qps": 0.0,
        "peak_ram_mb": 0.0,
        "disk_size_mb": 0.0,
        "reopen_time_s": 0.0,
        "timeout_policy_s": 60,
        "status": "success",
        "host_fingerprint": {},
        "raw_data_path": "raw.parquet",
    }


@pytest.mark.unit
def test_summary_new_fields_default_to_none() -> None:
    # Arrange + Act
    summary = BenchmarkSummary(**_minimal_summary_kwargs())  # type: ignore[arg-type]

    # Assert
    assert summary.concurrency is None
    assert summary.delete_time_s is None
    assert summary.upsert_time_s is None
    assert summary.post_delete_recall is None


@pytest.mark.unit
def test_summary_accepts_new_fields() -> None:
    # Arrange + Act
    summary = BenchmarkSummary(
        **_minimal_summary_kwargs(),  # type: ignore[arg-type]
        concurrency=4,
        delete_time_s=0.5,
        upsert_time_s=0.25,
        post_delete_recall=0.9,
    )

    # Assert
    assert summary.concurrency == 4
    assert summary.delete_time_s == 0.5
    assert summary.upsert_time_s == 0.25
    assert summary.post_delete_recall == 0.9


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("concurrency", 0),
        ("delete_time_s", -1.0),
        ("upsert_time_s", -1.0),
        ("post_delete_recall", 1.5),
    ],
    ids=["concurrency-zero", "delete-negative", "upsert-negative", "recall-out-of-range"],
)
def test_summary_rejects_invalid_new_field_values(
    field_name: str,
    invalid_value: float,
) -> None:
    # Arrange + Act + Assert
    with pytest.raises(ValidationError):
        BenchmarkSummary(
            **_minimal_summary_kwargs(),  # type: ignore[arg-type]
            **{field_name: invalid_value},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_path_lines", "match"),
    [
        ("num_partitions = 0", "num_partitions"),
        ("num_sub_vectors = 0", "num_sub_vectors"),
        ("nprobes = 0", "nprobes"),
        ("refine_factor = -1", "refine_factor"),
    ],
    ids=["partitions-zero", "subvectors-zero", "nprobes-zero", "refine-negative"],
)
def test_lancedb_config_rejects_out_of_domain_values(
    config_path_lines: str,
    match: str,
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "lancedb.toml"
    config_path.write_text(
        f"[engines.lancedb]\n{config_path_lines}\n",
        encoding="utf-8",
    )

    # Act + Assert
    with pytest.raises(ValidationError, match=match):
        load_config(config_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path_lines",
    [
        "search_ef = 0",
        "max_points = 0",
        "max_c2_queries = 0",
    ],
    ids=["chroma-ef-zero", "sweep-maxpoints-zero", "c2-budget-zero"],
)
def test_positive_fields_reject_zero(config_path_lines: str, tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "bounds.toml"
    config_path.write_text(config_path_lines + "\n", encoding="utf-8")

    # Act + Assert
    with pytest.raises(ValidationError):
        load_config(config_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("grid_name", "invalid_value"),
    [
        ("faiss_nprobe", 0),
        ("lancedb_nprobes", -3),
        ("lancedb_refine_factor", -1),
    ],
    ids=["nprobe-zero", "nprobes-negative", "refine-negative"],
)
def test_param_sweep_rejects_invalid_grid_values(
    grid_name: str,
    invalid_value: int,
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        f"[workloads.param_sweep]\n{grid_name} = [{invalid_value}]\n",
        encoding="utf-8",
    )

    # Act + Assert
    with pytest.raises(ValidationError):
        load_config(config_path)


@pytest.mark.unit
def test_raw_record_new_fields_default_to_none() -> None:
    # Arrange + Act
    record = RawRecord(
        run_id="run-test",
        engine="faiss",
        workload="a2",
        phase="query",
        status="success",
        timestamp_utc="2026-09-06T00:00:00Z",
    )

    # Assert
    assert record.concurrency_level is None
    assert record.search_params is None
    assert record.dataset_size is None


@pytest.mark.unit
def test_raw_record_accepts_new_fields() -> None:
    # Arrange + Act
    record = RawRecord(
        run_id="run-test",
        engine="lancedb",
        workload="sweep",
        phase="query",
        status="success",
        timestamp_utc="2026-09-06T00:00:00Z",
        concurrency_level=8,
        search_params={"nprobes": 4, "refine_factor": 1},
        dataset_size=3000,
    )

    # Assert
    assert record.concurrency_level == 8
    assert record.search_params == {"nprobes": 4, "refine_factor": 1}
    assert record.dataset_size == 3000


@pytest.mark.unit
def test_raw_record_rejects_invalid_concurrency_level() -> None:
    # Arrange + Act + Assert
    with pytest.raises(ValidationError):
        RawRecord(
            run_id="run-test",
            engine="faiss",
            workload="a2",
            phase="query",
            status="success",
            timestamp_utc="2026-09-06T00:00:00Z",
            concurrency_level=0,
        )


@pytest.mark.unit
def test_turbovec_config_defaults() -> None:
    # Arrange + Act
    config = BenchmarkConfig()

    # Assert: single-index defaults (4-bit, no TQ+ calibration) and the
    # legacy default engine set is untouched (l2 + no turbovec stays valid).
    assert config.engines.turbovec.bit_width == 4
    assert config.engines.turbovec.calibrate is False
    assert "turbovec" not in config.engines.enabled
    assert config.engines.distance_metric == "l2"


@pytest.mark.unit
def test_turbovec_config_rejects_out_of_domain_values() -> None:
    # Arrange + Act + Assert: bit_width is fixed by the library to {2, 3, 4}
    # and extra="forbid" keeps rejecting unknown keys (match pins the reason).
    with pytest.raises(ValidationError, match="bit_width"):
        TurboVecConfig(bit_width=5)
    with pytest.raises(ValidationError, match="unknown_key"):
        TurboVecConfig(bit_width=4, unknown_key=True)


@pytest.mark.unit
def test_turbovec_engine_name_accepted(tmp_path: Path) -> None:
    # Arrange: turbovec requires cosine, so the config must set it.
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        """
[engines]
enabled = ["turbovec"]
distance_metric = "cosine"

[engines.turbovec]
bit_width = 3
calibrate = true
""".strip(),
        encoding="utf-8",
    )

    # Act
    config = load_config(config_path)

    # Assert
    assert config.engines.enabled == ["turbovec"]
    assert config.engines.distance_metric == "cosine"
    assert config.engines.turbovec.bit_width == 3
    assert config.engines.turbovec.calibrate is True


@pytest.mark.unit
def test_turbovec_with_l2_rejected_at_config_load(tmp_path: Path) -> None:
    # Arrange: turbovec under l2 must fail at load time, not mid-campaign
    # inside the adapter (adversarial-review blast-radius fix).
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        """
[engines]
enabled = ["turbovec"]
""".strip(),
        encoding="utf-8",
    )

    # Act + Assert
    with pytest.raises(ValidationError, match="inner-product-only"):
        load_config(config_path)


@pytest.mark.unit
def test_turbovec_absent_keeps_l2_valid(tmp_path: Path) -> None:
    # Arrange: the default profile (no turbovec, l2) must keep loading.
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        """
[engines]
enabled = ["faiss", "chromadb", "lancedb"]
distance_metric = "l2"
""".strip(),
        encoding="utf-8",
    )

    # Act
    config = load_config(config_path)

    # Assert
    assert config.engines.distance_metric == "l2"


@pytest.mark.unit
def test_param_sweep_turbovec_bit_width_grid(tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        """
[workloads.param_sweep]
turbovec_bit_width = [2, 4]
""".strip(),
        encoding="utf-8",
    )

    # Act
    config = load_config(config_path)

    # Assert
    assert config.workloads.param_sweep.turbovec_bit_width == [2, 4]


@pytest.mark.parametrize("bad_width", [5, -1, 0], ids=["over", "negative", "zero"])
@pytest.mark.unit
def test_param_sweep_rejects_invalid_turbovec_widths(bad_width: int, tmp_path: Path) -> None:
    # Arrange: only {2, 3, 4} are legal bit widths.
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        f"""
[workloads.param_sweep]
turbovec_bit_width = [{bad_width}]
""".strip(),
        encoding="utf-8",
    )

    # Act + Assert
    with pytest.raises(ValidationError, match="turbovec_bit_width"):
        load_config(config_path)


@pytest.mark.unit
def test_impact_profile_loads_with_five_engine_cosine_campaign() -> None:
    # Arrange + Act: the committed impact profile is the 5-engine cosine
    # campaign — it must parse end-to-end with the turbovec schema.
    config = load_config(REPO_ROOT / "configs" / "impact-benchmark.toml")

    # Assert
    assert config.engines.enabled == [
        "faiss",
        "chromadb",
        "lancedb",
        "sqlite_vec",
        "turbovec",
    ]
    assert config.engines.distance_metric == "cosine"
    assert config.engines.turbovec.bit_width == 4
    assert config.engines.turbovec.calibrate is True
    assert config.workloads.param_sweep.turbovec_bit_width == [2, 3, 4]
