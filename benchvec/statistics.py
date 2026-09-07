"""Statistical rigor helpers: bootstrap CIs, CV, Mann-Whitney U, and aggregation.

This module centralizes the statistics used by the benchmark expansion plan:

- :func:`bootstrap_ci` — percentile bootstrap confidence interval for any
  statistic (mean by default), reproducible through a seed.
- :func:`mann_whitney_u` — two-sided asymptotic Mann-Whitney U test.
- :func:`coefficient_of_variation` — cross-repetition dispersion measure.
- :func:`holm_adjust` — Holm step-down family-wise correction.
- :func:`aggregate_engine_stats` — consolidates raw per-query latencies into
  per-group summary statistics and Holm-corrected pairwise engine comparisons,
  returned as a pair of Arrow tables ready for parquet persistence.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
from scipy.stats import mannwhitneyu  # type: ignore[import-untyped]  # scipy-stubs absent

from benchvec.metrics import BenchmarkSummary, percentile

logger = logging.getLogger(__name__)

# Minimum samples per engine side before a pairwise comparison is trusted.
MIN_PAIRWISE_SAMPLES: int = 5

# Significance threshold applied to Holm-adjusted p-values.
SIGNIFICANCE_ALPHA: float = 0.05

# Explicit schemas keep the output column order/types stable even when a
# campaign yields no rows, so downstream parquet writers stay deterministic.
STATS_SUMMARY_SCHEMA = pa.schema(
    [
        ("workload", pa.string()),
        ("engine", pa.string()),
        ("k", pa.int64()),
        ("concurrency_level", pa.int64()),
        ("dataset_size", pa.int64()),
        ("n", pa.int64()),
        ("mean_latency_ms", pa.float64()),
        ("ci_low", pa.float64()),
        ("ci_high", pa.float64()),
        ("p95_latency_ms", pa.float64()),
        ("cv_latency", pa.float64()),
        ("mean_recall", pa.float64()),
    ]
)

STATS_PAIRWISE_SCHEMA = pa.schema(
    [
        ("workload", pa.string()),
        ("k", pa.int64()),
        ("concurrency_level", pa.int64()),
        ("dataset_size", pa.int64()),
        ("engine_a", pa.string()),
        ("engine_b", pa.string()),
        ("delta_mean_pct", pa.float64()),
        ("u_statistic", pa.float64()),
        ("p_value", pa.float64()),
        ("holm_adjusted_p", pa.float64()),
        ("significant_at_0_05", pa.bool_()),
    ]
)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval for ``statistic``.

    Steps:
    1. Validate inputs (non-empty sample, valid ``alpha`` and ``n_boot``).
    2. Resample the data with replacement ``n_boot`` times using
       ``np.random.default_rng(seed)`` so results are reproducible.
    3. Apply ``statistic`` to every resample.
    4. Return the ``alpha/2`` and ``1 - alpha/2`` percentiles of the bootstrap
       distribution (percentile method).

    Args:
        values: Sample observations.
        statistic: Statistic applied to each resample (defaults to the mean).
        n_boot: Number of bootstrap resamples (must be positive).
        alpha: Significance level; the interval covers ``1 - alpha`` (0 < alpha < 1).
        seed: Seed for the resampling RNG (reproducibility contract).

    Returns:
        Tuple ``(ci_low, ci_high)`` with the bootstrap interval bounds.

    Raises:
        ValueError: If ``values`` is empty, ``alpha`` is outside (0, 1),
            or ``n_boot`` is not positive.
    """
    # Step 1: validate inputs before touching numpy.
    if len(values) == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in the open interval (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be a positive integer")

    # Step 2: resample with replacement, one iteration at a time to keep
    # memory bounded for large query-per-group sample sizes.
    data = np.asarray(values, dtype=float)
    sample_size = data.shape[0]
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        # Draw one resample of the same size as the original sample.
        resample = data[rng.integers(0, sample_size, size=sample_size)]
        # Step 3: apply the requested statistic (list() keeps the callable
        # contract Sequence[float]-typed and works for statistics.fmean).
        boot_stats[iteration] = statistic(resample.tolist())

    # Step 4: percentile method interval from the bootstrap distribution.
    lower_bound, upper_bound = np.quantile(boot_stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lower_bound), float(upper_bound))


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Run a two-sided Mann-Whitney U test between two independent samples.

    Wraps :func:`scipy.stats.mannwhitneyu` with ``alternative="two-sided"``,
    ``method="asymptotic"`` and ``use_continuity=True`` — the right default
    for the hundreds of per-query latencies collected per engine.

    Args:
        a: First sample (e.g. engine A per-query latencies).
        b: Second sample (e.g. engine B per-query latencies).

    Returns:
        Tuple ``(u_statistic, p_value)``; ``(nan, nan)`` when either sample
        is empty (degradation instead of crash).
    """
    # Degrade gracefully on empty input: log and return NaN sentinels.
    if len(a) == 0 or len(b) == 0:
        logger.warning("mann_whitney_u received an empty sample; returning NaN results")
        return (float("nan"), float("nan"))

    # Delegate the rank computation and tie handling to scipy.
    result = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    return (float(result.statistic), float(result.pvalue))


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Compute the population coefficient of variation (pstdev / mean).

    Args:
        values: Sample observations (e.g. latencies across repetitions).

    Returns:
        ``pstdev(values) / mean(values)``; ``0.0`` for fewer than two values
        or when the mean is zero (CV is undefined in both cases).
    """
    # Fewer than two values carry no dispersion information.
    if len(values) < 2:
        return 0.0

    mean_value = statistics.fmean(values)
    # Guard the division: a zero mean makes the ratio undefined.
    if mean_value == 0:
        return 0.0

    return statistics.pstdev(values) / mean_value


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply the Holm step-down correction to a family of p-values.

    Steps:
    1. Sort the p-values ascending.
    2. Scale each sorted p-value by its decreasing multiplier ``(m - rank)``
       where ``m`` is the family size.
    3. Enforce monotonicity by carrying a running maximum (step-down).
    4. Cap results at ``1.0`` and restore the original input order.

    NaN entries are passed through as NaN and skipped in the running-maximum
    chain, so a single degenerate comparison cannot corrupt the family.

    Args:
        p_values: Raw p-values of one test family.

    Returns:
        Adjusted p-values in the same order as the input.
    """
    family_size = len(p_values)
    if family_size == 0:
        return []

    # NaN results (degenerate comparisons) keep their NaN adjusted value.
    adjusted = [float("nan")] * family_size
    finite_sorted = sorted(
        (p_value, index)
        for index, p_value in enumerate(p_values)
        if not math.isnan(p_value)
    )

    # Step-down: running maximum of (multiplier * sorted p-value).
    running_max = 0.0
    for rank, (p_value, index) in enumerate(finite_sorted):
        candidate = (family_size - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def _first_not_none(*values: Any) -> Any:
    """Return the first argument that is not ``None``, or ``None``."""

    for value in values:
        if value is not None:
            return value
    return None


def _ordering_value(value: Any) -> Any:
    """Map ``None`` to a sort sentinel so mixed keys order deterministically."""

    return -1 if value is None else value


def aggregate_engine_stats(
    summaries: Sequence[BenchmarkSummary],
    raw_query_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
) -> tuple[pa.Table, pa.Table]:
    """Aggregate raw per-query rows into summary and pairwise statistics tables.

    Steps:
    1. Build a ``run_id`` lookup from the consolidated summaries so raw rows
       that lack ``k``/``concurrency_level`` can still be attributed (raw
       parquet rows do not always carry those columns).
    2. Group raw rows by ``(workload, engine, k, concurrency_level)`` with
       None-tolerant keys — pooling different k or concurrency levels would
       mix distinct latency distributions (adversarial-review finding M3).
    3. Per group: sample count, mean latency with a seeded bootstrap CI,
       p95, pooled per-query CV, and mean recall.
    4. Per ``(workload, k, concurrency_level)`` family with at least two
       engines: all engine pairs (requiring ``MIN_PAIRWISE_SAMPLES`` per
       side) get a Mann-Whitney U test, a relative mean delta, and a Holm
       step-down correction applied within the family's pair set.

    Args:
        summaries: Consolidated benchmark summaries (used to enrich raw rows
            missing grouping fields via their ``run_id``).
        raw_query_rows: Raw per-query rows as mappings (e.g. from
            ``raw_store.load_raw_records``); ``latency_ms`` values of ``None``
            are skipped.
        seed: Seed forwarded to the bootstrap resampling.

    Returns:
        Tuple ``(stats_summary, stats_pairwise)`` of Arrow tables following
        ``STATS_SUMMARY_SCHEMA`` and ``STATS_PAIRWISE_SCHEMA``.
    """
    # Step 1: run_id -> grouping fields, for enriching raw rows that miss them.
    run_key_lookup: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        run_key_lookup[summary.run_id] = {
            "workload": summary.workload,
            "engine": summary.engine,
            "k": summary.k,
            "concurrency_level": summary.concurrency,
            "dataset_size": summary.dataset_size,
        }

    key_fields = ("workload", "engine", "k", "concurrency_level", "dataset_size")

    # Step 2: group raw rows by the full None-tolerant key.
    groups: dict[tuple[Any, Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_query_rows:
        # Rows without a string run_id simply get no summary enrichment.
        run_id = row.get("run_id")
        fallback = run_key_lookup.get(run_id, {}) if isinstance(run_id, str) else {}
        group_key = tuple(
            _first_not_none(row.get(field), fallback.get(field)) for field in key_fields
        )
        groups[group_key].append(row)

    # Steps 3 + 4: per-group aggregates and the pairwise family map.
    summary_rows: list[dict[str, Any]] = []
    family_engines: dict[
        tuple[Any, Any, Any, Any], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for (workload, engine, k, concurrency, dataset_size), rows in groups.items():
        # Collect latency values, skipping None measurements.
        latencies = [
            float(row["latency_ms"])
            for row in rows
            if row.get("latency_ms") is not None
        ]
        # A group without any latency measurement cannot be summarized.
        if not latencies:
            continue

        recalls = [
            float(row["recall_at_k"])
            for row in rows
            if row.get("recall_at_k") is not None
        ]
        ci_low, ci_high = bootstrap_ci(latencies, seed=seed)
        summary_rows.append(
            {
                "workload": workload,
                "engine": engine,
                "k": k,
                "concurrency_level": concurrency,
                "dataset_size": dataset_size,
                "n": len(latencies),
                "mean_latency_ms": statistics.fmean(latencies),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p95_latency_ms": percentile(latencies, 0.95),
                "cv_latency": coefficient_of_variation(latencies),
                "mean_recall": statistics.fmean(recalls) if recalls else None,
            }
        )
        # Register the group's latencies for its pairwise family. The sweep
        # workload is excluded from pairwise tests: each engine runs a
        # different parameter grid, so pooled latencies are not comparable.
        if workload != "sweep":
            family_engines[(workload, k, concurrency, dataset_size)][engine].extend(
                latencies
            )

    # Deterministic ordering: None keys sort behind the -1 sentinel.
    summary_rows.sort(
        key=lambda row: (
            str(row["workload"] or ""),
            str(row["engine"] or ""),
            _ordering_value(row["k"]),
            _ordering_value(row["concurrency_level"]),
        )
    )

    # Step 4: pairwise comparisons with Holm correction per family.
    pairwise_rows: list[dict[str, Any]] = []
    for family_key in sorted(
        family_engines,
        key=lambda key: (
            str(key[0] or ""),
            _ordering_value(key[1]),
            _ordering_value(key[2]),
            _ordering_value(key[3]),
        ),
    ):
        workload, k, concurrency, dataset_size = family_key
        engines_map = family_engines[family_key]
        eligible_pairs: list[dict[str, Any]] = []
        raw_p_values: list[float] = []

        for engine_a, engine_b in combinations(sorted(engines_map), 2):
            latencies_a = engines_map[engine_a]
            latencies_b = engines_map[engine_b]
            # Skip pairs without enough samples on either side.
            if (
                len(latencies_a) < MIN_PAIRWISE_SAMPLES
                or len(latencies_b) < MIN_PAIRWISE_SAMPLES
            ):
                logger.debug(
                    "Skipping pair (%s, %s) for workload %s: insufficient samples",
                    engine_a,
                    engine_b,
                    workload,
                )
                continue

            mean_a = statistics.fmean(latencies_a)
            mean_b = statistics.fmean(latencies_b)
            # Relative delta is undefined against a zero-mean baseline.
            delta_mean_pct: float | None = (
                (mean_b - mean_a) / mean_a * 100.0 if mean_a != 0 else None
            )
            u_statistic, p_value = mann_whitney_u(latencies_a, latencies_b)
            eligible_pairs.append(
                {
                    "engine_a": engine_a,
                    "engine_b": engine_b,
                    "delta_mean_pct": delta_mean_pct,
                    "u_statistic": u_statistic,
                    "p_value": p_value,
                }
            )
            raw_p_values.append(p_value)

        # Holm correction across this family's pair set only.
        adjusted_p_values = holm_adjust(raw_p_values)
        for pair, adjusted_p in zip(eligible_pairs, adjusted_p_values):
            pairwise_rows.append(
                {
                    "workload": workload,
                    "k": k,
                    "concurrency_level": concurrency,
                    "dataset_size": dataset_size,
                    "engine_a": pair["engine_a"],
                    "engine_b": pair["engine_b"],
                    "delta_mean_pct": pair["delta_mean_pct"],
                    "u_statistic": pair["u_statistic"],
                    "p_value": pair["p_value"],
                    "holm_adjusted_p": adjusted_p,
                    # NaN adjusted p-values compare False: never significant.
                    "significant_at_0_05": bool(adjusted_p < SIGNIFICANCE_ALPHA),
                }
            )

    # Build both tables with explicit schemas for stable output types.
    summary_table = pa.Table.from_pylist(summary_rows, schema=STATS_SUMMARY_SCHEMA)
    pairwise_table = pa.Table.from_pylist(pairwise_rows, schema=STATS_PAIRWISE_SCHEMA)
    return (summary_table, pairwise_table)
