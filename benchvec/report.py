"""Summary consolidation and run inspection utilities for BenchVec."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

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

_WORKLOAD_ORDER = ["a1", "a2", "b1", "b2", "b3", "b4"]
_ENGINE_ORDER = ["faiss", "chromadb", "lancedb"]


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


def generate_charts(
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    project: Optional[Project] = None,
) -> list[Path]:
    """Generate 6 comparative benchmark charts and save them as PNG files.

    Charts produced:
    1. Latency comparison (mean / p95 / p99) per engine × workload
    2. QPS / throughput per engine × workload
    3. Recall@K per engine × workload
    4. Peak RAM (MB) per engine × workload
    5. Disk footprint (MB) per engine (aggregated across workloads)
    6. Build time / Time-to-serve per engine (aggregated)
    
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

    if project:
        logger.info(f"Generated {len(generated)} charts for project to {output_dir}")

    return generated
