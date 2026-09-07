#!/usr/bin/env python3
"""
generate_analysis.py — Deterministic Benchmark Analysis Report Generator

Reads consolidated Parquet results from a BenchVec project and produces a
structured ANALYSIS_REPORT.md markdown file in the project directory.

Standalone analysis generator for consolidated benchmark results.
Usage:
    python generate_analysis.py --project-dir <path> [--dataset-name <name>]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from enum import Enum
from dataclasses import dataclass



# ---------------------------------------------------------------------------
# CONFIGURATION CLASSES
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Output format for the analysis report."""
    markdown = "markdown"
    html = "html"
    json = "json"


class ReportType(str, Enum):
    """Type of report to generate."""
    executive_summary = "executive_summary"
    detailed = "detailed"
    statistical_only = "statistical_only"


@dataclass
class AnalysisConfig:
    """Configuration for analysis report generation."""
    input_dir: Path
    output_path: Path
    output_format: OutputFormat = OutputFormat.markdown
    report_type: ReportType = ReportType.detailed
    include_charts: bool = True
    dataset_name: str | None = None
    suppress_sections: list[str] | None = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ANALYSIS_REPORT.md from BenchVec project results"
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=Path,
        help="Path to the benchmark project directory (e.g. .benchmarks/projects/benchmark_default)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Human-readable dataset name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output path for the markdown report (default: {project_dir}/ANALYSIS_REPORT.md)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


NUMERIC_COLS: list[str] = [
    "mean_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "std_latency_ms",
    "qps",
    "recall_at_k",
    "peak_ram_mb",
    "disk_size_mb",
    "build_time_s",
    "load_time_s",
    "reopen_time_s",
]

WORKLOAD_LABELS: dict[str, str] = {
    "a1": "A1 — Ingestion & Index Build",
    "a2": "A2 — Pure Vector Search (ANN)",
    "b1": "B1 — Filtered Search (Metadata)",
    "b2": "B2 — Persistent Streaming Ingestion",
    "b3": "B3 — Index Reopen After Persistence",
    "b4": "B4 — Memory Pressure",
    "c1": "C1 — Concurrent Search (Thread Ladder)",
    "c2": "C2 — Dataset Scaling Ladder",
    "d1": "D1 — Delete & Upsert Lifecycle",
    "sweep": "Sweep — Recall-QPS Parameter Sweep",
}


def load_results(project_dir: Path) -> pd.DataFrame:
    """Load consolidated results parquet, falling back to individual summaries."""
    parquet_path = project_dir / "reports" / "final.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # Fallback: glob individual summary files
    # Handle both cases: project_dir as results root, or project_dir/results
    summaries: list[dict[str, Any]] = []
    
    # Try glob pattern for results directory (project_dir IS the results dir)
    glob_patterns = [
        project_dir.glob("*/*/*/summary.parquet"),  # engine/workload/run-id pattern
        project_dir.glob("results/*/*/*/summary.parquet"),  # raw project dir pattern
    ]
    
    for pattern in glob_patterns:
        for summary_file in sorted(pattern):
            try:
                row = pd.read_parquet(summary_file)
                summaries.extend(row.to_dict(orient="records"))
            except Exception:  # noqa: BLE001
                continue
        if summaries:
            break  # Found results, stop trying patterns
    
    if not summaries:
        msg = f"No results found in {project_dir}"
        raise FileNotFoundError(msg)
    df = pd.DataFrame(summaries)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detect_dataset_name(df: pd.DataFrame, project_dir: Path) -> str:
    """Auto-detect human-readable dataset name from results or project name."""
    if df is not None and "dataset_name" in df.columns:
        raw = str(df["dataset_name"].dropna().iloc[0]) if len(df) > 0 else ""
        if "cohere" in raw.lower() or "wikipedia" in raw.lower():
            return "Cohere Wikipedia 22-12"
        if "sift" in raw.lower():
            return "SIFT1M"
        if "gist" in raw.lower():
            return "GIST1M"
    name = project_dir.name.lower()
    if "cohere" in name:
        return "Cohere Wikipedia 22-12"
    if "sift" in name:
        return "SIFT1M"
    if "gist" in name:
        return "GIST1M"
    return project_dir.name


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_ms(value: float | None) -> str:
    """Format millisecond value with 4 decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def fmt_float(value: float | None, decimals: int = 2) -> str:
    """Format float with given decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_recall(value: float | None) -> str:
    """Format recall as percentage."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f} ({value * 100:.2f}%)"


def fmt_qps(value: float | None) -> str:
    """Format QPS with thousand separators."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.1f}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def build_executive_summary(df: pd.DataFrame, dataset_name: str) -> str:
    """Build executive summary section."""
    engines = sorted(df["engine"].unique())
    workloads = sorted(df["workload"].unique())
    total_runs = len(df)
    n_engines = len(engines)
    n_workloads = len(workloads)

    lines: list[str] = [
        "## 1. Executive Summary\n",
        f"This report presents a deterministic analysis of the **{dataset_name}** benchmark results",
        f"from {total_runs} experimental runs across {n_engines} vector database engines",
        f"and {n_workloads} workload type(s).\n",
        f"- **Engines evaluated:** {', '.join(e.upper() for e in engines)}",
        f"- **Workloads executed:** {', '.join(w.upper() for w in workloads)}",
        f"- **Total runs:** {total_runs}",
        f"- **Generated at:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Best engine per metric for A2 (search benchmark - most comparable)
    a2 = df[df["workload"] == "a2"]
    if not a2.empty:
        eng_avg = a2.groupby("engine")[["mean_latency_ms", "qps", "recall_at_k"]].mean()
        if "mean_latency_ms" in eng_avg.columns:
            best_latency = eng_avg["mean_latency_ms"].idxmin()
            lines.append(f"- **Fastest engine (A2 latency):** {best_latency.upper()}")
        if "qps" in eng_avg.columns:
            best_qps = eng_avg["qps"].idxmax()
            lines.append(f"- **Highest throughput (A2 QPS):** {best_qps.upper()}")
        if "recall_at_k" in eng_avg.columns:
            best_recall = eng_avg["recall_at_k"].idxmax()
            lines.append(f"- **Best recall (A2):** {best_recall.upper()}")

    lines.append("")
    return "\n".join(lines)


def build_engine_comparison_table(df: pd.DataFrame) -> str:
    """Build overall engine comparison table."""
    lines: list[str] = [
        "## 2. Overall Engine Comparison\n",
        "Average metrics across all workloads:\n",
        "| Engine | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Avg QPS | Avg Recall | Peak RAM (MB) | Disk (MB) |",
        "|--------|-----------------|-----------------|-----------------|---------|-----------|--------------|-----------|",
    ]

    numeric = [c for c in ["mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "qps", "recall_at_k", "peak_ram_mb", "disk_size_mb"] if c in df.columns]
    eng_avg = df.groupby("engine")[numeric].mean()

    for engine in sorted(eng_avg.index):
        row = eng_avg.loc[engine]
        lines.append(
            f"| **{engine.upper()}** "
            f"| {fmt_ms(row.get('mean_latency_ms'))} "
            f"| {fmt_ms(row.get('p95_latency_ms'))} "
            f"| {fmt_ms(row.get('p99_latency_ms'))} "
            f"| {fmt_qps(row.get('qps'))} "
            f"| {fmt_recall(row.get('recall_at_k'))} "
            f"| {fmt_float(row.get('peak_ram_mb'))} "
            f"| {fmt_float(row.get('disk_size_mb'))} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_workload_breakdown(df: pd.DataFrame) -> str:
    """Build per-workload performance breakdown."""
    lines: list[str] = ["## 3. Performance by Workload\n"]

    numeric = [c for c in ["mean_latency_ms", "p95_latency_ms", "qps", "recall_at_k", "peak_ram_mb", "disk_size_mb", "build_time_s"] if c in df.columns]

    for workload in sorted(df["workload"].unique()):
        label = WORKLOAD_LABELS.get(workload, workload.upper())
        wdf = df[df["workload"] == workload]
        eng_avg = wdf.groupby("engine")[numeric].mean()

        lines.append(f"### {label}\n")
        lines.append("| Engine | Mean Latency (ms) | P95 Latency (ms) | QPS | Recall@K | Peak RAM (MB) | Disk (MB) |")
        lines.append("|--------|-----------------|-----------------|-----|---------|--------------|-----------|")

        for engine in sorted(eng_avg.index):
            row = eng_avg.loc[engine]
            lines.append(
                f"| **{engine.upper()}** "
                f"| {fmt_ms(row.get('mean_latency_ms'))} "
                f"| {fmt_ms(row.get('p95_latency_ms'))} "
                f"| {fmt_qps(row.get('qps'))} "
                f"| {fmt_recall(row.get('recall_at_k'))} "
                f"| {fmt_float(row.get('peak_ram_mb'))} "
                f"| {fmt_float(row.get('disk_size_mb'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_statistical_analysis(df: pd.DataFrame) -> str:
    """Build statistical analysis with std deviation."""
    lines: list[str] = [
        "## 4. Statistical Analysis\n",
        "Latency statistics for search workloads (A2, B1, B2, B3, B4):\n",
    ]

    search_workloads = [w for w in df["workload"].unique() if w != "a1"]
    search_df = df[df["workload"].isin(search_workloads)]

    if search_df.empty:
        lines.append("_No search workload data available._\n")
        return "\n".join(lines)

    numeric = [c for c in ["mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "std_latency_ms", "qps"] if c in df.columns]
    stats = search_df.groupby("engine")[numeric].agg(["mean", "std"]).round(4)

    lines.append("| Engine | Mean Latency | ±Std | P95 | P99 | Mean QPS |")
    lines.append("|--------|-------------|------|-----|-----|---------|")

    for engine in sorted(df["engine"].unique()):
        if engine not in stats.index:
            continue
        row = stats.loc[engine]
        mean_lat = fmt_ms(row.get(("mean_latency_ms", "mean")))
        std_lat = fmt_ms(row.get(("std_latency_ms", "mean")))
        p95 = fmt_ms(row.get(("p95_latency_ms", "mean")))
        p99 = fmt_ms(row.get(("p99_latency_ms", "mean")))
        qps_val = fmt_qps(row.get(("qps", "mean")))
        lines.append(f"| **{engine.upper()}** | {mean_lat} | {std_lat} | {p95} | {p99} | {qps_val} |")

    lines.append("")
    return "\n".join(lines)


def build_significance_section(project_dir: Path) -> str:
    """Build the pairwise significance section from stats_pairwise.parquet.

    Values come from benchvec.statistics (Mann-Whitney U with Holm step-down
    correction within each workload family); the section is omitted when the
    parquet is absent so legacy projects keep their previous output.
    """

    import pyarrow.parquet as pq

    project_dir_path = Path(project_dir)
    candidates = [
        project_dir_path / "stats_pairwise.parquet",
        project_dir_path / "reports" / "stats_pairwise.parquet",
        # The CLI points input_dir at <project>/results; reports live one up.
        project_dir_path.parent / "reports" / "stats_pairwise.parquet",
    ]
    pairwise_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if pairwise_path is None:
        return ""
    try:
        rows = pq.read_table(pairwise_path).to_pylist()
    except Exception:
        return ""
    if not rows:
        return ""

    lines: list[str] = [
        "## Statistical Significance (Pairwise Engine Comparisons)\n",
        "Mann-Whitney U on per-query latencies; Holm step-down correction "
        "applied within each workload family (alpha = 0.05).\n",
        "| Workload | k | Concurrency | Dataset Size | Engine A | Engine B | ΔMean % | p-value | Holm-adj. p | Significant |",
        "|----------|---|-------------|--------------|----------|----------|---------|---------|-------------|-------------|",
    ]
    for row in rows[:100]:
        delta = row.get("delta_mean_pct")
        delta_text = f"{delta:+.1f}%" if delta is not None else "n/a"
        p_value = row.get("p_value")
        p_text = f"{p_value:.2e}" if p_value is not None and p_value == p_value else "n/a"
        holm_p = row.get("holm_adjusted_p")
        holm_text = f"{holm_p:.2e}" if holm_p is not None and holm_p == holm_p else "n/a"
        significant = row.get("significant_at_0_05")
        lines.append(
            f"| {row.get('workload')} "
            f"| {row.get('k') if row.get('k') is not None else '-'} "
            f"| {row.get('concurrency_level') if row.get('concurrency_level') is not None else '-'} "
            f"| {row.get('dataset_size') if row.get('dataset_size') is not None else '-'} "
            f"| {row.get('engine_a')} | {row.get('engine_b')} "
            f"| {delta_text} | {p_text} | {holm_text} "
            f"| {'yes' if significant else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_key_findings(df: pd.DataFrame, dataset_name: str) -> str:
    """Generate key findings from the benchmark data."""
    lines: list[str] = ["## 5. Key Findings\n"]

    engines = sorted(df["engine"].unique())
    workloads = sorted(df["workload"].unique())

    numeric_cols = [c for c in ["mean_latency_ms", "qps", "recall_at_k", "peak_ram_mb", "disk_size_mb"] if c in df.columns]
    eng_avg = df.groupby("engine")[numeric_cols].mean()

    findings: list[str] = []

    # Latency champion
    if "mean_latency_ms" in eng_avg.columns:
        fastest = eng_avg["mean_latency_ms"].idxmin()
        slowest = eng_avg["mean_latency_ms"].idxmax()
        ratio = eng_avg["mean_latency_ms"][slowest] / eng_avg["mean_latency_ms"][fastest]
        findings.append(
            f"**{fastest.upper()}** achieves the lowest average latency "
            f"({fmt_ms(eng_avg['mean_latency_ms'][fastest])} ms), "
            f"{ratio:.1f}× faster than {slowest.upper()}."
        )

    # QPS champion
    if "qps" in eng_avg.columns:
        highest_qps = eng_avg["qps"].idxmax()
        findings.append(
            f"**{highest_qps.upper()}** delivers the highest throughput "
            f"({fmt_qps(eng_avg['qps'][highest_qps])} QPS)."
        )

    # Recall
    if "recall_at_k" in eng_avg.columns:
        perfect_recall = [e for e in engines if abs(eng_avg.loc[e, "recall_at_k"] - 1.0) < 0.001] if engines else []
        if perfect_recall:
            findings.append(
                f"**{', '.join(e.upper() for e in perfect_recall)}** achieve perfect Recall@K (1.0000), "
                "confirming exact search fidelity."
            )
        approx_engines = [e for e in engines if e not in perfect_recall and e in eng_avg.index]
        for e in approx_engines:
            r = eng_avg.loc[e, "recall_at_k"]
            if not pd.isna(r):
                findings.append(
                    f"**{e.upper()}** uses approximate search (HNSW), achieving Recall@K = {r:.4f} "
                    f"({r * 100:.2f}%)."
                )

    # Memory
    if "peak_ram_mb" in eng_avg.columns:
        lightest = eng_avg["peak_ram_mb"].idxmin()
        findings.append(
            f"**{lightest.upper()}** has the lowest peak RAM usage "
            f"({fmt_float(eng_avg['peak_ram_mb'][lightest])} MB)."
        )

    # Disk
    if "disk_size_mb" in eng_avg.columns:
        smallest_disk = eng_avg["disk_size_mb"].idxmin()
        findings.append(
            f"**{smallest_disk.upper()}** has the smallest disk footprint "
            f"({fmt_float(eng_avg['disk_size_mb'][smallest_disk])} MB)."
        )

    for i, finding in enumerate(findings, 1):
        lines.append(f"{i}. {finding}")

    lines.append("")
    return "\n".join(lines)


def build_methodology_section(df: pd.DataFrame, dataset_name: str) -> str:
    """Build methodology section."""
    engines = sorted(df["engine"].unique())
    workloads = sorted(df["workload"].unique())
    n_reps = int(df.get("n_repetitions", pd.Series([5])).dropna().max()) if "n_repetitions" in df.columns else 5
    dims = df["dimension"].dropna().iloc[0] if "dimension" in df.columns and not df["dimension"].dropna().empty else "N/A"
    dataset_size = df["dataset_size"].dropna().iloc[0] if "dataset_size" in df.columns and not df["dataset_size"].dropna().empty else "N/A"
    distance = df["distance_metric"].dropna().iloc[0] if "distance_metric" in df.columns and not df["distance_metric"].dropna().empty else "N/A"
    index_type = df["index_type"].dropna().iloc[0] if "index_type" in df.columns and not df["index_type"].dropna().empty else "N/A"

    lines: list[str] = [
        "## 6. Methodology\n",
        f"- **Dataset:** {dataset_name}",
        f"- **Dataset size:** {dataset_size} vectors × {dims} dimensions",
        f"- **Distance metric:** {distance}",
        f"- **Engines:** {', '.join(e.upper() for e in engines)}",
        f"- **FAISS index type:** {index_type}",
        f"- **Workloads:** {', '.join(w.upper() for w in workloads)}",
        f"- **Repetitions per workload:** {n_reps}",
        "- **Warmup queries:** 20 per run",
        "- **Deterministic sampling:** seed=42",
        "- **Ground truth:** exact nearest neighbors computed via brute-force FAISS",
        "- **Hardware fingerprint:** recorded per run in env.json",
        "",
    ]
    return "\n".join(lines)


def build_charts_section(project_dir: Path) -> str:
    """Build charts reference section."""
    charts_dir = project_dir / "reports" / "charts"
    chart_labels = {
        "chart_01_latency.png": "Mean / P95 / P99 Latency Comparison by Engine",
        "chart_02_qps.png": "Throughput (QPS) Comparison by Engine",
        "chart_03_recall.png": "Recall@K by Engine",
        "chart_04_peak_ram.png": "Peak Memory Usage (MB) by Engine",
        "chart_05_disk_footprint.png": "Disk Footprint (MB) by Engine",
        "chart_06_build_time.png": "Build Time + Time-to-Serve by Engine",
    }

    lines: list[str] = ["## 7. Charts\n"]
    if not charts_dir.exists():
        lines.append("_Charts not yet generated. Run `main.py charts --project <name>` first._\n")
        return "\n".join(lines)

    for chart_file, label in chart_labels.items():
        chart_path = charts_dir / chart_file
        if chart_path.exists():
            rel_path = chart_path.relative_to(project_dir)
            lines.append(f"### {label}\n")
            lines.append(f"![{label}]({rel_path})\n")
        else:
            lines.append(f"- `{chart_file}`: _not found_")

    lines.append("")
    return "\n".join(lines)


def build_config_appendix(project_dir: Path) -> str:
    """Build configuration appendix."""
    lines: list[str] = ["## 8. Configuration & Environment\n"]

    config_path = project_dir / "config.toml"
    if config_path.exists():
        lines.append("### Project Configuration\n")
        lines.append("```toml")
        lines.append(config_path.read_text(encoding="utf-8").strip())
        lines.append("```\n")

    # Try to find one env.json
    env_files = sorted(project_dir.glob("results/*/*/*/env.json"))
    if env_files:
        try:
            env = json.loads(env_files[0].read_text(encoding="utf-8"))
            lines.append("### Hardware Environment (first run)\n")
            lines.append("```json")
            lines.append(json.dumps(env, indent=2, default=str))
            lines.append("```\n")
        except Exception:  # noqa: BLE001
            pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_report(config: AnalysisConfig) -> Path:
    """Generate comprehensive analysis report from benchmark results.
    
    Args:
        config: AnalysisConfig with analysis parameters
        
    Returns:
        Path to the generated report file
    """
    # Determine output path
    output_path = config.output_path
    
    # Determine dataset name
    dataset_name = config.dataset_name or detect_dataset_name(None, config.input_dir)
    
    # Load results
    project_dir = config.input_dir / "results" if (config.input_dir / "results").exists() else config.input_dir
    df = load_results(project_dir)
    
    if dataset_name is None:
        dataset_name = detect_dataset_name(df, project_dir)
    
    # Build report sections
    sections = []
    
    if config.report_type in (ReportType.executive_summary, ReportType.detailed):
        sections.append(build_executive_summary(df, dataset_name))
    
    if config.report_type == ReportType.detailed:
        sections.append(build_engine_comparison_table(df))
        sections.append(build_workload_breakdown(df))
        sections.append(build_key_findings(df, dataset_name))
    
    if config.report_type in (ReportType.statistical_only, ReportType.detailed):
        sections.append(build_statistical_analysis(df))
    
    if config.report_type == ReportType.detailed:
        significance_section = build_significance_section(project_dir)
        if significance_section:
            sections.append(significance_section)
    
    if config.report_type in (ReportType.executive_summary, ReportType.detailed):
        sections.append(build_methodology_section(df, dataset_name))
        if config.include_charts:
            sections.append(build_charts_section(project_dir))
        sections.append(build_config_appendix(project_dir))
    
    # Generate output based on format
    if config.output_format == OutputFormat.markdown:
        content = "\n\n".join(sections)
    elif config.output_format == OutputFormat.json:
        content = json.dumps({
            "dataset": dataset_name,
            "report_type": config.report_type.value,
            "sections": sections,
            "generated_at": datetime.datetime.now().isoformat(),
        }, indent=2)
    elif config.output_format == OutputFormat.html:
        # Simple HTML wrapper
        content = f"<html><head><title>Analysis Report</title></head><body>" +                   "".join(f"<section>{s}</section>" for s in sections) +                   "</body></html>"
    else:
        content = "\n\n".join(sections)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    
    return output_path



def main() -> None:
    """Entry point."""
    """Entry point."""
    args = parse_args()
    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    config = AnalysisConfig(
        input_dir=project_dir,
        output_path=Path(args.output),
        output_format=OutputFormat.markdown,
        report_type=ReportType.detailed,
        dataset_name=args.dataset_name,
    )
    output_path = generate_report(config)
    print(f"Report generated: {output_path}")
if __name__ == "__main__":
    main()
