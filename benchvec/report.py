"""Summary consolidation and run inspection utilities for BenchVec."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from benchvec.metrics import BenchmarkSummary, load_summary
from benchvec.project import Project
from benchvec.raw_store import load_raw_records

logger = logging.getLogger(__name__)


def collect_summary_paths(results_dir: str | Path) -> list[Path]:
    """Collect every persisted summary file under the results tree."""

    base_dir = Path(results_dir)
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("summary.parquet"))


def consolidate_summaries(
    results_dir: str | Path | None = None,
    *,
    project: Optional[Project] = None,
) -> list[BenchmarkSummary]:
    """Load all persisted summaries into memory.
    
    Args:
        results_dir: Optional path to results directory. If None and project is provided,
                    uses project.results_dir. If both None, raises ValueError.
        project: Optional Project instance to resolve results_dir from.
    
    Returns:
        List of loaded BenchmarkSummary instances.
    
    Raises:
        ValueError: If both results_dir and project are None.
    """
    if results_dir is None and project is None:
        msg = "Either results_dir or project must be provided"
        raise ValueError(msg)

    if results_dir is None and project is not None:
        results_dir = project.results_dir
        logger.debug(f"Using project results directory: {results_dir}")

    return [load_summary(path) for path in collect_summary_paths(results_dir)]


def summaries_to_table(summaries: list[BenchmarkSummary]) -> pa.Table:
    """Convert benchmark summaries into an Arrow table."""

    payload = [summary.model_dump(mode="json") for summary in summaries]
    return pa.Table.from_pylist(payload)


def write_consolidated_report(
    results_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    project: Optional[Project] = None,
) -> Path:
    """Write a consolidated report to CSV or Parquet.
    
    Args:
        results_dir: Optional path to results directory. If None and project is provided,
                    uses project.results_dir.
        output_path: Optional output file path. If None and project is provided,
                    uses project.reports_dir / "final_results.parquet".
        project: Optional Project instance to resolve directories from.
    
    Returns:
        Path to the written report file.
    
    Raises:
        ValueError: If results_dir and project are both None, or if output_path and
                   project are both None.
    """
    if results_dir is None and project is None:
        msg = "Either results_dir or project must be provided"
        raise ValueError(msg)

    if results_dir is None:
        results_dir = project.results_dir
        logger.debug(f"Using project results directory: {results_dir}")

    if output_path is None and project is None:
        msg = "Either output_path or project must be provided"
        raise ValueError(msg)

    if output_path is None:
        output_path = project.reports_dir / "final_results.parquet"
        logger.debug(f"Using project reports directory for output: {output_path}")

    summaries = consolidate_summaries(results_dir, project=project)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not summaries:
        frame = pd.DataFrame()
    else:
        frame = pd.DataFrame([summary.model_dump(mode="json") for summary in summaries])

    if target_path.suffix.lower() == ".csv":
        frame.to_csv(target_path, index=False)
    else:
        # Parquet cannot represent empty struct columns (e.g., {}).
        # Normalize nested containers to deterministic JSON strings.
        for column in frame.columns:
            if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
        frame.to_parquet(target_path, index=False)

    if project:
        logger.info(f"Consolidated report written to project: {target_path}")

    return target_path


def write_results_index(
    results_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    project: Optional[Project] = None,
) -> Path:
    """Write an index relating run IDs to persisted artifacts.
    
    Args:
        results_dir: Optional path to results directory. If None and project is provided,
                    uses project.results_dir.
        output_path: Optional output file path. If None and project is provided,
                    uses project.reports_dir / "results_index.parquet".
        project: Optional Project instance to resolve directories from.
    
    Returns:
        Path to the written index file.
    
    Raises:
        ValueError: If results_dir and project are both None, or if output_path and
                   project are both None.
    """
    if results_dir is None and project is None:
        msg = "Either results_dir or project must be provided"
        raise ValueError(msg)

    if results_dir is None:
        results_dir = project.results_dir
        logger.debug(f"Using project results directory: {results_dir}")

    if output_path is None and project is None:
        msg = "Either output_path or project must be provided"
        raise ValueError(msg)

    if output_path is None:
        output_path = project.reports_dir / "results_index.parquet"
        logger.debug(f"Using project reports directory for index: {output_path}")

    rows: list[dict[str, Any]] = []
    for summary_path in collect_summary_paths(results_dir):
        run_dir = summary_path.parent
        summary = load_summary(summary_path)
        rows.append(
            {
                "run_id": summary.run_id,
                "engine": summary.engine,
                "workload": summary.workload,
                "summary_path": str(summary_path),
                "raw_path": str(run_dir / summary.raw_data_path),
                "env_path": str(run_dir / "env.json"),
            }
        )
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target_path)

    if project:
        logger.info(f"Results index written to project: {target_path}")

    return target_path


def find_run_directory(results_dir: str | Path, run_id: str) -> Path:
    """Locate one run directory by run ID."""

    base_dir = Path(results_dir)
    for summary_path in collect_summary_paths(base_dir):
        summary = load_summary(summary_path)
        if summary.run_id == run_id:
            return summary_path.parent
    raise FileNotFoundError(f"Run ID not found: {run_id}")


def inspect_run(
    results_dir: str | Path | None = None,
    run_id: str | None = None,
    *,
    project: Optional[Project] = None,
) -> dict[str, Any]:
    """Load summary, environment snapshot, and raw micro-data for one run.
    
    Args:
        results_dir: Optional path to results directory. If None and project is provided,
                    uses project.results_dir.
        run_id: Run ID to inspect (required).
        project: Optional Project instance to resolve results_dir from.
    
    Returns:
        Dictionary with keys: run_dir, summary, env, raw_records.
    
    Raises:
        ValueError: If results_dir and project are both None, or run_id is None.
    """
    if results_dir is None and project is None:
        msg = "Either results_dir or project must be provided"
        raise ValueError(msg)

    if run_id is None:
        msg = "run_id parameter is required"
        raise ValueError(msg)

    if results_dir is None:
        results_dir = project.results_dir
        logger.debug(f"Using project results directory: {results_dir}")

    run_dir = find_run_directory(results_dir, run_id)
    summary = load_summary(run_dir / "summary.parquet")
    env_payload = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    raw_records = load_raw_records(run_dir)

    if project:
        logger.debug(f"Inspected run {run_id} from project: {project.name}")

    return {
        "run_dir": run_dir,
        "summary": summary,
        "env": env_payload,
        "raw_records": raw_records,
    }


# ---------------------------------------------------------------------------
# Comparative chart generation
# ---------------------------------------------------------------------------

_CHART_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

_WORKLOAD_ORDER = ["a1", "a2", "b1", "b2", "b3", "b4", "c1", "c2", "d1", "sweep"]
_ENGINE_ORDER = ["faiss", "chromadb", "lancedb", "sqlite_vec", "turbovec"]


def _safe_df(
    results_dir: str | Path | None = None,
    *,
    project: Optional[Project] = None,
) -> pd.DataFrame:
    """Load consolidated summaries as a DataFrame, returning empty frame if none.
    
    Args:
        results_dir: Optional path to results directory. If None and project is provided,
                    uses project.results_dir.
        project: Optional Project instance to resolve results_dir from.
    
    Returns:
        DataFrame containing consolidated summaries, or empty DataFrame if none found.
    """
    summaries = consolidate_summaries(results_dir, project=project)
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame([s.model_dump(mode="json") for s in summaries])


def pareto_mask(qps_values: Sequence[float], recall_values: Sequence[float]) -> list[bool]:
    """Return the Pareto-optimal subset (no other point beats both axes).

    A point is dominated when some other point has strictly higher QPS and
    recall. Pure helper so chart 08's frontier is unit-testable.
    """

    mask: list[bool] = []
    for index, (qps, recall) in enumerate(zip(qps_values, recall_values)):
        dominated = any(
            other_qps > qps and other_recall > recall
            for other_index, (other_qps, other_recall) in enumerate(zip(qps_values, recall_values))
            if other_index != index
        )
        mask.append(not dominated)
    return mask


def collect_workload_raw_rows(results_dir: str | Path, workload: str) -> list[dict[str, Any]]:
    """Collect raw query rows for one workload across every run directory."""

    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return rows
    for parquet_path in sorted(results_path.glob("**/raw_queries_*.parquet")):
        try:
            table = pq.read_table(parquet_path)
        except Exception:  # pragma: no cover - unreadable partial files
            continue
        for row in table.to_pylist():
            if row.get("workload") == workload:
                rows.append(row)
    return rows


def write_stats_report(
    results_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Compute and persist the statistical report (CI + significance).

    Writes ``stats_summary.parquet`` (per workload × engine × k ×
    concurrency: bootstrap CI, CV, mean recall) and ``stats_pairwise.parquet``
    (pairwise Mann-Whitney with Holm step-down correction) using
    ``benchvec.statistics.aggregate_engine_stats``.
    """

    from benchvec.statistics import STATS_PAIRWISE_SCHEMA, STATS_SUMMARY_SCHEMA, aggregate_engine_stats

    summaries = consolidate_summaries(results_dir)
    raw_rows: list[dict[str, Any]] = []
    for workload in ("a2", "b1", "b2", "b3", "b4", "c1", "c2", "d1", "sweep"):
        raw_rows.extend(collect_workload_raw_rows(results_dir, workload))
    stats_summary, stats_pairwise = aggregate_engine_stats(summaries, raw_rows, seed=seed)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "stats_summary.parquet"
    pairwise_path = target_dir / "stats_pairwise.parquet"
    pq.write_table(stats_summary.cast(STATS_SUMMARY_SCHEMA), summary_path)
    pq.write_table(stats_pairwise.cast(STATS_PAIRWISE_SCHEMA), pairwise_path)
    logger.info(
        "stats report written: %d summary rows, %d pairwise rows",
        stats_summary.num_rows,
        stats_pairwise.num_rows,
    )
    return summary_path, pairwise_path


def generate_charts(
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    project: Optional[Project] = None,
) -> list[Path]:
    """Generate the comparative benchmark charts and save them as PNG files.

    Charts produced:
    1. Latency comparison (mean / p95 / p99) per engine × workload
    2. QPS / throughput per engine × workload
    3. Recall@K per engine × workload
    4. Peak RAM (MB) per engine × workload
    5. Disk footprint (MB) per engine (aggregated across workloads)
    6. Build time / Time-to-serve per engine (aggregated)
    7. Concurrency: QPS / latency per thread level (C1)
    8. Pareto frontier: recall vs QPS per search-params point (sweep)
    9. Scaling: latency / recall per dataset size (C2)
    10. k-sweep: Recall@K per k value (A2)
    
    Args:
        results_dir: Optional path to results directory. If None and project is provided,
                    uses project.results_dir.
        output_dir: Optional path to output directory. If None and project is provided,
                   uses project.reports_dir / "charts".
        project: Optional Project instance to resolve directories from.

    Returns:
        List of paths for the generated PNG files.
    
    Raises:
        ValueError: If results_dir and project are both None, or if output_dir and
                   project are both None.
    """
    if results_dir is None and project is None:
        msg = "Either results_dir or project must be provided"
        raise ValueError(msg)

    if results_dir is None:
        results_dir = project.results_dir
        logger.debug(f"Using project results directory: {results_dir}")

    if output_dir is None and project is None:
        msg = "Either output_dir or project must be provided"
        raise ValueError(msg)

    if output_dir is None:
        output_dir = project.reports_dir / "charts"
        logger.debug(f"Using project charts directory for output: {output_dir}")

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    matplotlib.use("Agg")  # non-interactive backend – safe in all environments
    sns.set_theme(style="whitegrid", palette=_CHART_PALETTE)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _safe_df(results_dir)
    if df.empty:
        return []

    # Normalise engine / workload order to known lists where present
    if "engine" in df.columns:
        known = [e for e in _ENGINE_ORDER if e in df["engine"].unique()]
        extra = [e for e in df["engine"].unique() if e not in _ENGINE_ORDER]
        df["engine"] = pd.Categorical(df["engine"], categories=known + extra, ordered=True)
    if "workload" in df.columns:
        known_w = [w for w in _WORKLOAD_ORDER if w in df["workload"].unique()]
        extra_w = [w for w in df["workload"].unique() if w not in _WORKLOAD_ORDER]
        df["workload"] = pd.Categorical(
            df["workload"], categories=known_w + extra_w, ordered=True
        )

    generated: list[Path] = []

    # Keep the full frame for chart 10 (the k-sweep view needs every k);
    # charts 01-03 below use the k-filtered frame.
    full_df = df
    # k-sweep support: charts 01-03 must not mix A2 rows across k levels
    # (their latencies/recalls differ systematically), so keep only the
    # largest k when several are present.
    if "workload" in df.columns and "k" in df.columns:
        a2_mask = df["workload"] == "a2"
        a2_ks = df.loc[a2_mask, "k"].dropna()
        if a2_ks.nunique() > 1:
            max_k = a2_ks.max()
            df = pd.concat(
                [df[~a2_mask], df[a2_mask & (df["k"] == max_k)]],
                ignore_index=True,
            )

    # ------------------------------------------------------------------
    # Chart 1 — Latency: mean / p95 / p99
    # ------------------------------------------------------------------
    latency_df = df[df["mean_latency_ms"].notna() & (df["mean_latency_ms"] > 0)].copy()
    if not latency_df.empty:
        melted = latency_df.melt(
            id_vars=["engine", "workload"],
            value_vars=["mean_latency_ms", "p95_latency_ms", "p99_latency_ms"],
            var_name="metric",
            value_name="latency_ms",
        )
        melted["metric"] = melted["metric"].map(
            {"mean_latency_ms": "Mean", "p95_latency_ms": "p95", "p99_latency_ms": "p99"}
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=melted,
            x="workload",
            y="latency_ms",
            hue="engine",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title("Latency Comparison by Workload (ms)", fontsize=14)
        ax.set_xlabel("Workload")
        ax.set_ylabel("Latency (ms)")
        ax.legend(title="Engine")
        fig.tight_layout()
        path1 = out_dir / "chart_01_latency.png"
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        generated.append(path1)

    # ------------------------------------------------------------------
    # Chart 2 — QPS / Throughput per engine × workload
    # ------------------------------------------------------------------
    qps_df = df[df["qps"].notna() & (df["qps"] > 0)].copy()
    if not qps_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=qps_df,
            x="workload",
            y="qps",
            hue="engine",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title("QPS (Queries / Vectors per Second) by Workload", fontsize=14)
        ax.set_xlabel("Workload")
        ax.set_ylabel("QPS")
        ax.legend(title="Engine")
        fig.tight_layout()
        path2 = out_dir / "chart_02_qps.png"
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        generated.append(path2)

    # ------------------------------------------------------------------
    # Chart 3 — Recall@K per engine × workload
    # ------------------------------------------------------------------
    recall_df = df[df["recall_at_k"].notna()].copy()
    if not recall_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=recall_df,
            x="workload",
            y="recall_at_k",
            hue="engine",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Recall@K by Engine and Workload", fontsize=14)
        ax.set_xlabel("Workload")
        ax.set_ylabel("Recall@K")
        ax.legend(title="Engine")
        fig.tight_layout()
        path3 = out_dir / "chart_03_recall.png"
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        generated.append(path3)

    # ------------------------------------------------------------------
    # Chart 4 — Peak RAM (MB) per engine × workload
    # ------------------------------------------------------------------
    ram_df = df[df["peak_ram_mb"].notna() & (df["peak_ram_mb"] > 0)].copy()
    if not ram_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=ram_df,
            x="workload",
            y="peak_ram_mb",
            hue="engine",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title("Peak RAM Usage by Engine and Workload (MB)", fontsize=14)
        ax.set_xlabel("Workload")
        ax.set_ylabel("Peak RAM (MB)")
        ax.legend(title="Engine")
        fig.tight_layout()
        path4 = out_dir / "chart_04_peak_ram.png"
        fig.savefig(path4, dpi=150)
        plt.close(fig)
        generated.append(path4)

    # ------------------------------------------------------------------
    # Chart 5 — Disk footprint (MB) per engine (aggregated median)
    # ------------------------------------------------------------------
    disk_df = df[df["disk_size_mb"].notna() & (df["disk_size_mb"] > 0)].copy()
    if not disk_df.empty:
        agg_disk = (
            disk_df.groupby("engine", observed=True)["disk_size_mb"]
            .median()
            .reset_index()
            .sort_values("disk_size_mb", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=agg_disk,
            x="engine",
            y="disk_size_mb",
            palette=_CHART_PALETTE[: len(agg_disk)],
            ax=ax,
        )
        ax.set_title("Disk Footprint by Engine (Median across Workloads, MB)", fontsize=14)
        ax.set_xlabel("Engine")
        ax.set_ylabel("Disk Size (MB)")
        fig.tight_layout()
        path5 = out_dir / "chart_05_disk_footprint.png"
        fig.savefig(path5, dpi=150)
        plt.close(fig)
        generated.append(path5)

    # ------------------------------------------------------------------
    # Chart 6 — Build time / Time-to-serve per engine (median)
    # ------------------------------------------------------------------
    build_df = df[df["build_time_s"].notna()].copy()
    if not build_df.empty:
        agg_build = (
            build_df.groupby("engine", observed=True)[["build_time_s", "time_to_serve_s"]]
            .median()
            .reset_index()
        )
        melted_build = agg_build.melt(
            id_vars="engine",
            value_vars=["build_time_s", "time_to_serve_s"],
            var_name="phase",
            value_name="seconds",
        )
        melted_build["phase"] = melted_build["phase"].map(
            {"build_time_s": "Build Time", "time_to_serve_s": "Time to Serve"}
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=melted_build,
            y="engine",
            x="seconds",
            hue="phase",
            palette=_CHART_PALETTE[:2],
            orient="h",
            ax=ax,
        )
        ax.set_title("Build Time and Time-to-Serve by Engine (Median, seconds)", fontsize=14)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Engine")
        ax.legend(title="Phase")
        fig.tight_layout()
        path6 = out_dir / "chart_06_build_time.png"
        fig.savefig(path6, dpi=150)
        plt.close(fig)
        generated.append(path6)

    # ------------------------------------------------------------------
    # Chart 7 — Concurrency: QPS and p95 latency vs thread count (C1)
    # ------------------------------------------------------------------
    c1_df = df[(df["workload"] == "c1") & df["concurrency"].notna()].copy()
    if not c1_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for engine_name, engine_group in c1_df.groupby("engine", observed=True):
            grouped = engine_group.groupby("concurrency", observed=True).median(numeric_only=True)
            axes[0].plot(grouped.index, grouped["qps"], marker="o", label=str(engine_name))
            axes[1].plot(grouped.index, grouped["p95_latency_ms"], marker="o", label=str(engine_name))
        axes[0].set_title("QPS vs Concurrent Threads (C1)")
        axes[0].set_xlabel("Threads")
        axes[0].set_ylabel("QPS")
        axes[1].set_title("p95 Latency vs Concurrent Threads (C1)")
        axes[1].set_xlabel("Threads")
        axes[1].set_ylabel("p95 Latency (ms)")
        for axis in axes:
            axis.legend(title="Engine")
        fig.tight_layout()
        path7 = out_dir / "chart_07_concurrency.png"
        fig.savefig(path7, dpi=150)
        plt.close(fig)
        generated.append(path7)

    # ------------------------------------------------------------------
    # Chart 8 — Pareto frontier: Recall vs QPS per search-params point
    # ------------------------------------------------------------------
    sweep_rows = collect_workload_raw_rows(results_dir, "sweep")
    sweep_points = [
        {
            "engine": row.get("engine"),
            "search_params": json.dumps(row.get("search_params"), sort_keys=True)
            if row.get("search_params")
            else "{}",
            "latency_ms": latency,
            "recall_at_k": recall,
        }
        for row in sweep_rows
        for latency, recall in [
            (
                row.get("latency_ms"),
                row.get("recall_at_k"),
            )
        ]
        if latency is not None and recall is not None
    ]
    if sweep_points:
        sweep_df = pd.DataFrame(sweep_points)
        point_stats = (
            sweep_df.groupby(["engine", "search_params"])
            .agg(mean_recall=("recall_at_k", "mean"), total_latency=("latency_ms", "sum"), n=("latency_ms", "size"))
            .reset_index()
        )
        point_stats["qps"] = point_stats["n"] / (point_stats["total_latency"] / 1000.0)
        fig, ax = plt.subplots(figsize=(10, 7))
        for engine_name, engine_group in point_stats.groupby("engine"):
            ordered = engine_group.sort_values("qps")
            frontier = pareto_mask(
                ordered["qps"].tolist(),
                ordered["mean_recall"].tolist(),
            )
            ax.plot(ordered["qps"], ordered["mean_recall"], marker="o", alpha=0.4, label=None)
            frontier_points = ordered[frontier]
            ax.plot(
                frontier_points["qps"],
                frontier_points["mean_recall"],
                marker="o",
                linewidth=2,
                label=str(engine_name),
            )
        ax.set_xscale("log")
        ax.set_title("Recall vs QPS Frontier per Search-Params Point (Sweep)")
        ax.set_xlabel("QPS (log scale)")
        ax.set_ylabel("Mean Recall@K")
        ax.legend(title="Engine (Pareto frontier)")
        fig.tight_layout()
        path8 = out_dir / "chart_08_pareto.png"
        fig.savefig(path8, dpi=150)
        plt.close(fig)
        generated.append(path8)

    # ------------------------------------------------------------------
    # Chart 9 — Scaling: latency and ingest throughput vs dataset size (C2)
    # ------------------------------------------------------------------
    c2_df = df[(df["workload"] == "c2") & df["dataset_size"].notna()].copy()
    if not c2_df.empty:
        c2_df["ingest_vps"] = c2_df.apply(
            lambda row: row["dataset_size"] / row["load_time_s"]
            if row["load_time_s"] and row["load_time_s"] > 0
            else 0.0,
            axis=1,
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for engine_name, engine_group in c2_df.groupby("engine", observed=True):
            grouped = engine_group.groupby("dataset_size").median(numeric_only=True)
            axes[0].plot(grouped.index, grouped["mean_latency_ms"], marker="o", label=str(engine_name))
            axes[1].plot(grouped.index, grouped["ingest_vps"], marker="o", label=str(engine_name))
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("Dataset Size (vectors, log)")
            axis.legend(title="Engine")
        axes[0].set_title("Mean Latency vs Dataset Size (C2)")
        axes[0].set_ylabel("Mean Latency (ms)")
        axes[1].set_title("Ingest Throughput vs Dataset Size (C2)")
        axes[1].set_ylabel("Vectors / Second")
        fig.tight_layout()
        path9 = out_dir / "chart_09_scaling.png"
        fig.savefig(path9, dpi=150)
        plt.close(fig)
        generated.append(path9)

    # ------------------------------------------------------------------
    # Chart 10 — k-sweep: recall and latency vs k (A2 summaries)
    # ------------------------------------------------------------------
    a2_all = full_df[full_df["workload"] == "a2"].copy()
    if not a2_all.empty and a2_all["k"].nunique() > 1:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax2 = ax1.twinx()
        for engine_name, engine_group in a2_all.groupby("engine", observed=True):
            grouped = engine_group.groupby("k").mean(numeric_only=True)
            ax1.bar(
                grouped.index,
                grouped["recall_at_k"],
                width=(grouped.index.max() - grouped.index.min()) / (len(grouped) * 3 + 1),
                alpha=0.5,
                label=f"{engine_name} recall",
            )
            ax2.plot(grouped.index, grouped["mean_latency_ms"], marker="s", label=f"{engine_name} latency")
        ax1.set_xscale("log")
        ax1.set_xlabel("k (log scale)")
        ax1.set_ylabel("Recall@K")
        ax1.set_ylim(0.0, 1.05)
        ax2.set_ylabel("Mean Latency (ms)")
        ax1.set_title("Recall and Latency vs k (A2 k-Sweep)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
        fig.tight_layout()
        path10 = out_dir / "chart_10_k_sweep.png"
        fig.savefig(path10, dpi=150)
        plt.close(fig)
        generated.append(path10)

    if project:
        logger.info(f"Generated {len(generated)} charts for project to {output_dir}")

    return generated
