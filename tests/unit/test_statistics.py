"""Unit tests for the statistics module added by the expansion plan."""

from __future__ import annotations

import logging
import math

import numpy as np
import pyarrow as pa
import pytest
from pytest import LogCaptureFixture

from benchvec.metrics import BenchmarkSummary
from benchvec.statistics import (
    STATS_PAIRWISE_SCHEMA,
    STATS_SUMMARY_SCHEMA,
    aggregate_engine_stats,
    bootstrap_ci,
    coefficient_of_variation,
    holm_adjust,
    mann_whitney_u,
)

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_ci_contains_sample_mean() -> None:
    # Arrange: deterministic sample with mean exactly 10.0.
    values: list[float] = [8.0, 12.0] * 25

    # Act
    ci_low, ci_high = bootstrap_ci(values, seed=42)

    # Assert: the interval brackets the sample mean and is well-formed.
    assert ci_low <= 10.0 <= ci_high
    assert 0.0 < ci_low < ci_high


@pytest.mark.unit
def test_bootstrap_ci_width_shrinks_with_larger_samples() -> None:
    # Arrange: two samples from the same distribution, different sizes.
    rng = np.random.default_rng(7)
    small_sample = rng.normal(loc=100.0, scale=10.0, size=25).tolist()
    large_sample = rng.normal(loc=100.0, scale=10.0, size=500).tolist()

    # Act
    small_low, small_high = bootstrap_ci(small_sample, seed=42)
    large_low, large_high = bootstrap_ci(large_sample, seed=42)

    # Assert: the CI half-width scales ~1/sqrt(n), so it must shrink.
    assert (small_high - small_low) > (large_high - large_low)


@pytest.mark.unit
def test_bootstrap_ci_is_deterministic_with_same_seed() -> None:
    # Arrange: fixed sample and two identical calls, plus a different seed.
    values: list[float] = [10.0, 12.0, 11.0, 13.0, 10.5, 12.5, 11.5, 14.0] * 4

    # Act
    first = bootstrap_ci(values, seed=42)
    second = bootstrap_ci(values, seed=42)
    other_seed = bootstrap_ci(values, seed=43)

    # Assert: same seed reproduces the interval exactly; a different seed
    # yields a different resampling path for this data.
    assert first == second
    assert other_seed != first


@pytest.mark.unit
def test_bootstrap_ci_empty_input_raises_value_error() -> None:
    # Arrange + Act + Assert
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([])


@pytest.mark.unit
def test_bootstrap_ci_supports_custom_statistic() -> None:
    # Arrange: use max as the bootstrap statistic.
    values: list[float] = [1.0 + float(i) for i in range(50)]

    # Act
    ci_low, ci_high = bootstrap_ci(values, statistic=max, seed=42)

    # Assert: bootstrap maxima can never exceed the sample maximum.
    assert ci_low < ci_high
    assert ci_high <= max(values)


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mann_whitney_u_same_distribution_not_significant() -> None:
    # Arrange: two samples from the same distribution.
    rng = np.random.default_rng(11)
    sample_a = rng.normal(loc=100.0, scale=10.0, size=120).tolist()
    sample_b = rng.normal(loc=100.0, scale=10.0, size=120).tolist()

    # Act
    u_statistic, p_value = mann_whitney_u(sample_a, sample_b)

    # Assert: no detectable difference at alpha = 0.05.
    assert u_statistic >= 0.0
    assert p_value > 0.05


@pytest.mark.unit
def test_mann_whitney_u_shifted_distribution_significant() -> None:
    # Arrange: second sample shifted upward by 50% of the mean.
    rng = np.random.default_rng(11)
    sample_a = rng.normal(loc=100.0, scale=10.0, size=120).tolist()
    sample_b = rng.normal(loc=150.0, scale=10.0, size=120).tolist()

    # Act
    _u_statistic, p_value = mann_whitney_u(sample_a, sample_b)

    # Assert: the +50% shift must be detected.
    assert p_value < 0.05


@pytest.mark.unit
def test_mann_whitney_u_empty_sample_returns_nan_and_warns(
    caplog: LogCaptureFixture,
) -> None:
    # Arrange: capture warnings from the statistics module logger.
    with caplog.at_level(logging.WARNING, logger="benchvec.statistics"):
        # Act: one empty side, then both sides empty.
        u_one_empty, p_one_empty = mann_whitney_u([], [1.0, 2.0, 3.0])
        u_both_empty, p_both_empty = mann_whitney_u([], [])

    # Assert: NaN sentinels instead of a crash, plus a logged warning.
    assert math.isnan(u_one_empty)
    assert math.isnan(p_one_empty)
    assert math.isnan(u_both_empty)
    assert math.isnan(p_both_empty)
    assert any("empty sample" in record.getMessage() for record in caplog.records)


@pytest.mark.unit
def test_mann_whitney_u_identical_constant_samples_do_not_crash() -> None:
    # Arrange: fully tied constant samples (degenerate variance case).
    # Act
    u_statistic, p_value = mann_whitney_u([5.0] * 6, [5.0] * 6)

    # Assert: scipy reports "no evidence of difference", not an exception.
    assert u_statistic == pytest.approx(18.0)
    assert p_value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# coefficient_of_variation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coefficient_of_variation_constant_series_is_zero() -> None:
    # Arrange + Act + Assert: zero dispersion stays zero.
    assert coefficient_of_variation([1.0, 1.0, 1.0, 1.0]) == 0.0


@pytest.mark.unit
def test_coefficient_of_variation_known_value() -> None:
    # Arrange: [1, 2, 3] has pstdev sqrt(2/3) and mean 2 -> sqrt(1/6).
    # Act + Assert
    assert coefficient_of_variation([1.0, 2.0, 3.0]) == pytest.approx(math.sqrt(1 / 6))


@pytest.mark.unit
def test_coefficient_of_variation_degenerate_inputs_return_zero() -> None:
    # Arrange + Act + Assert: empty, single value, and zero-mean inputs.
    assert coefficient_of_variation([]) == 0.0
    assert coefficient_of_variation([4.0]) == 0.0
    assert coefficient_of_variation([-2.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# holm_adjust
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_holm_adjust_hand_computed_example() -> None:
    # Arrange: p-values [0.01, 0.04, 0.03], m = 3.
    # Hand computation (sorted ascending: 0.01, 0.03, 0.04):
    #   rank 0: 3 * 0.01 = 0.03
    #   rank 1: max(0.03, 2 * 0.03) = 0.06
    #   rank 2: max(0.06, 1 * 0.04) = 0.06
    # Act + Assert: results mapped back to the original input order.
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.unit
def test_holm_adjust_single_value_unchanged() -> None:
    # Arrange + Act + Assert: multiplier m = 1 leaves the p-value as is.
    assert holm_adjust([0.03]) == pytest.approx([0.03])


@pytest.mark.unit
def test_holm_adjust_enforces_monotonicity() -> None:
    # Arrange: raw scaled values decrease (2*0.02 = 0.04 then 1*0.021 = 0.021).
    # Act + Assert: the step-down running maximum keeps results monotone.
    assert holm_adjust([0.02, 0.021]) == pytest.approx([0.04, 0.04])


@pytest.mark.unit
def test_holm_adjust_caps_at_one() -> None:
    # Arrange: 2 * 0.8 = 1.6 exceeds 1.0.
    # Act + Assert: results are capped at 1.0.
    assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])


@pytest.mark.unit
def test_holm_adjust_empty_family_returns_empty_list() -> None:
    # Arrange + Act + Assert
    assert holm_adjust([]) == []


# ---------------------------------------------------------------------------
# aggregate_engine_stats
# ---------------------------------------------------------------------------


def _summary_kwargs(**overrides: object) -> dict[str, object]:
    """Return the minimum kwargs BenchmarkSummary requires (helper pattern
    copied from tests/unit/test_schema_extensions.py)."""

    kwargs: dict[str, object] = {
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
    kwargs.update(overrides)
    return kwargs


def _raw_row(**overrides: object) -> dict[str, object]:
    """Return one synthetic raw query row with sensible defaults."""

    row: dict[str, object] = {
        "run_id": "run-1",
        "workload": "a2",
        "engine": "faiss",
        "k": 5,
        "concurrency_level": 1,
        "latency_ms": 10.0,
        "recall_at_k": None,
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_aggregate_summary_groups_separate_k_and_concurrency() -> None:
    # Arrange: same workload/engine across a k x concurrency grid; None
    # latencies are skipped and a None-only group disappears entirely.
    raw_rows = [
        _raw_row(k=5, concurrency_level=1, latency_ms=10.0, recall_at_k=0.9),
        _raw_row(k=5, concurrency_level=1, latency_ms=10.0, recall_at_k=1.0),
        _raw_row(k=5, concurrency_level=1, latency_ms=None),
        _raw_row(k=10, concurrency_level=1, latency_ms=20.0),
        _raw_row(k=10, concurrency_level=1, latency_ms=20.0),
        _raw_row(k=5, concurrency_level=8, latency_ms=30.0),
        _raw_row(k=5, concurrency_level=8, latency_ms=30.0),
        _raw_row(k=10, concurrency_level=8, latency_ms=40.0),
        _raw_row(k=7, concurrency_level=1, latency_ms=None),
    ]

    # Act
    summary_table, _pairwise_table = aggregate_engine_stats([], raw_rows, seed=42)

    # Assert: four distinct groups (M3: k and concurrency must not pool).
    assert summary_table.num_rows == 4
    stats_by_key = {
        (row["k"], row["concurrency_level"]): row
        for row in summary_table.to_pylist()
    }
    assert set(stats_by_key) == {(5, 1), (10, 1), (5, 8), (10, 8)}

    # None latency rows are excluded from n; values are exact.
    assert stats_by_key[(5, 1)]["n"] == 2
    assert stats_by_key[(5, 1)]["mean_latency_ms"] == pytest.approx(10.0)
    assert stats_by_key[(5, 1)]["mean_recall"] == pytest.approx(0.95)
    assert stats_by_key[(10, 1)]["mean_latency_ms"] == pytest.approx(20.0)
    assert stats_by_key[(10, 1)]["mean_recall"] is None
    assert stats_by_key[(5, 8)]["n"] == 2
    assert stats_by_key[(10, 8)]["n"] == 1  # single sample degrades gracefully

    # The all-None group (k=7) produced no summary row.
    assert all(row["k"] != 7 for row in summary_table.to_pylist())


@pytest.mark.unit
def test_aggregate_enriches_missing_group_fields_from_summaries() -> None:
    # Arrange: raw rows without k/concurrency_level keys; summaries supply
    # the missing fields through the run_id lookup.
    raw_rows = [
        _raw_row(run_id="run-1", k=None, concurrency_level=None, latency_ms=10.0),
        _raw_row(run_id="run-1", k=None, concurrency_level=None, latency_ms=11.0),
        _raw_row(run_id="run-2", k=None, concurrency_level=None, latency_ms=20.0),
        # Row-level values take precedence when present (concurrency stays
        # at the row default of 1, k is overridden to 99).
        _raw_row(run_id="run-1", k=99, latency_ms=12.0),
    ]
    summaries = [
        BenchmarkSummary(**_summary_kwargs(run_id="run-1", k=10, concurrency=4)),  # type: ignore[arg-type]
        BenchmarkSummary(**_summary_kwargs(run_id="run-2", k=20, concurrency=8)),  # type: ignore[arg-type]
    ]

    # Act
    summary_table, pairwise_table = aggregate_engine_stats(summaries, raw_rows, seed=42)

    # Assert: three groups; enriched fields come from the matching summary.
    rows = summary_table.to_pylist()
    assert summary_table.num_rows == 3
    stats_by_key = {(row["k"], row["concurrency_level"]): row for row in rows}
    assert set(stats_by_key) == {(10, 4), (20, 8), (99, 1)}

    # One engine per family: no pairwise comparisons exist yet.
    assert pairwise_table.num_rows == 0


@pytest.mark.unit
def test_aggregate_pairwise_holm_verdicts() -> None:
    # Arrange: one family with two clearly separated engines and one
    # overlapping engine, plus a 3-sample engine below the minimum.
    alpha_values = [9.5, 9.8, 10.0, 10.2, 9.9, 10.1, 9.7, 10.8]  # mean 10.0
    beta_values = [19.5, 19.8, 20.0, 20.2, 19.9, 20.1, 19.7, 20.8]  # mean 20.0
    gamma_values = [10.1, 9.9, 10.2, 10.0, 9.8, 10.3, 10.1, 9.9]  # ~10.04
    tiny_values = [30.0, 30.1, 30.2]  # only 3 samples -> never paired

    raw_rows = [
        _raw_row(workload="w1", engine=engine, latency_ms=latency)
        for engine, values in (
            ("alpha", alpha_values),
            ("beta", beta_values),
            ("gamma", gamma_values),
            ("tiny", tiny_values),
        )
        for latency in values
    ]

    # Act
    _summary_table, pairwise_table = aggregate_engine_stats([], raw_rows, seed=42)

    # Assert: three eligible pairs only; "tiny" is excluded everywhere.
    pairs = pairwise_table.to_pylist()
    assert {(row["engine_a"], row["engine_b"]) for row in pairs} == {
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
    }
    assert all("tiny" not in (row["engine_a"], row["engine_b"]) for row in pairs)

    verdict_by_pair = {
        (row["engine_a"], row["engine_b"]): row for row in pairs
    }
    # The 2x mean gap is significant with a +100% delta.
    alpha_beta = verdict_by_pair[("alpha", "beta")]
    assert alpha_beta["delta_mean_pct"] == pytest.approx(100.0)
    assert alpha_beta["significant_at_0_05"] is True
    # The overlapping pair is not significant even before correction.
    alpha_gamma = verdict_by_pair[("alpha", "gamma")]
    assert alpha_gamma["significant_at_0_05"] is False
    assert alpha_gamma["p_value"] > 0.05
    # Beta separates from gamma too.
    assert verdict_by_pair[("beta", "gamma")]["significant_at_0_05"] is True

    # Holm is applied within the family (3 pairs): the smallest raw p-value
    # gets exactly the 3x multiplier, and adjusted >= raw everywhere.
    smallest = min(pairs, key=lambda row: row["p_value"])
    assert smallest["holm_adjusted_p"] == pytest.approx(3 * smallest["p_value"])
    assert all(
        row["holm_adjusted_p"] >= row["p_value"] and row["holm_adjusted_p"] <= 1.0
        for row in pairs
    )
    # Verdicts are driven by the adjusted p-value, not the raw one.
    assert all(
        row["significant_at_0_05"] == (row["holm_adjusted_p"] < 0.05) for row in pairs
    )


@pytest.mark.unit
def test_aggregate_tables_survive_pylist_roundtrip() -> None:
    # Arrange: run the pairwise scenario and grab both output tables.
    raw_rows = [
        _raw_row(workload="w1", engine=engine, latency_ms=latency)
        for engine, values in (
            ("alpha", [9.5, 9.8, 10.0, 10.2, 9.9, 10.1, 9.7, 10.8]),
            ("beta", [19.5, 19.8, 20.0, 20.2, 19.9, 20.1, 19.7, 20.8]),
        )
        for latency in values
    ]
    summary_table, pairwise_table = aggregate_engine_stats([], raw_rows, seed=42)

    # Act: parquet-style round-trip through to_pylist/from_pylist.
    summary_rt = pa.Table.from_pylist(
        summary_table.to_pylist(), schema=STATS_SUMMARY_SCHEMA
    )
    pairwise_rt = pa.Table.from_pylist(
        pairwise_table.to_pylist(), schema=STATS_PAIRWISE_SCHEMA
    )

    # Assert: values (and None slots) survive the round-trip unchanged.
    assert summary_rt.to_pylist() == summary_table.to_pylist()
    assert pairwise_rt.to_pylist() == pairwise_table.to_pylist()


@pytest.mark.unit
def test_aggregate_empty_inputs_return_empty_typed_tables() -> None:
    # Arrange + Act
    summary_table, pairwise_table = aggregate_engine_stats([], [])

    # Assert: zero rows but stable schemas for parquet writers.
    assert summary_table.num_rows == 0
    assert pairwise_table.num_rows == 0
    assert summary_table.column_names == STATS_SUMMARY_SCHEMA.names
    assert pairwise_table.column_names == STATS_PAIRWISE_SCHEMA.names


@pytest.mark.unit
def test_holm_adjust_passes_nan_through_without_corrupting_family() -> None:
    from benchvec.statistics import holm_adjust

    adjusted = holm_adjust([0.01, float("nan"), 0.04])

    # NaN counts toward the family size (conservative): m=3.
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] != adjusted[1]  # NaN stays NaN
    assert adjusted[2] == pytest.approx(0.08)
