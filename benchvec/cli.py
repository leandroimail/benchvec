"""CLI for running, inspecting, and reporting BenchVec benchmarks."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from benchvec.config import load_config
from benchvec.datasets import (
    PreparedRecord,
    download_dataset_from_config,
    prepare_dataset_from_config,
)
from benchvec.metrics import BenchmarkSummary
from benchvec.project import Project
from benchvec.report import (
    generate_charts,
    inspect_run,
    write_consolidated_report,
    write_results_index,
)
from benchvec.runner import BenchmarkRunner

logger = logging.getLogger(__name__)

app = click.Group(help="BenchVec reproducible benchmark CLI")
console = Console()
DEFAULT_WORKLOADS = ["a1", "a2", "b1", "b2", "b3", "b4"]


def _validate_project_name(name: str) -> bool:
    """Validate project name (alphanumeric + hyphen, underscore, dot).

    Args:
        name: Project name to validate

    Returns:
        True if valid, False otherwise
    """
    if not name:
        return False
    # Allow alphanumeric, hyphen, underscore, and dot
    pattern = r"^[a-zA-Z0-9_.-]+$"
    return bool(re.match(pattern, name))


def _print_summary_row(summary: BenchmarkSummary) -> None:
    """Print a formatted summary row.

    Args:
        summary: BenchmarkSummary instance to print
    """
    console.print(
        f"[{summary.status}] {summary.engine}/{summary.workload} run_id={summary.run_id} "
        f"latency={summary.mean_latency_ms:.3f}ms recall={summary.recall_at_k}"
    )


def _default_queries(
    runner: BenchmarkRunner,
    records: list[PreparedRecord],
) -> list[PreparedRecord]:
    """Get default query records from runner.

    Args:
        runner: BenchmarkRunner instance
        records: Prepared records

    Returns:
        List of query records
    """
    return runner.default_query_records(records)


def _get_or_create_project(project_name: str | None) -> Project:
    """Get existing project or create default project.

    Args:
        project_name: Project name, or None to use default

    Returns:
        Project instance

    Raises:
        click.Abort: If project name is invalid
    """
    name = project_name or "benchmark_default"
    try:
        return Project.from_name(name)
    except ValueError as e:
        console.print(f"[red]Error: Invalid project name '{name}': {e}[/red]")
        raise click.Abort()


def _find_config(project: Project, config_name: str | None = None) -> Path:
    """Find config file for the project.

    Lookup order:
    1. If config_name provided, look in project or configs/ directory
    2. Check project directory for any .toml/.yaml file
    3. Fall back to configs/ directory

    Args:
        project: Project instance
        config_name: Optional specific config filename

    Returns:
        Path to config file

    Raises:
        click.Abort: If no config found
    """
    try:
        return project.find_config(config_name)
    except FileNotFoundError as e:
        console.print(f"[red]Error finding config: {e}[/red]")
        raise click.Abort()


def _execute_run(
    config_path: Path,
    workloads: tuple[str, ...],
    output_dir: Path | None,
    project: Project | None = None,
) -> None:
    """Execute benchmark run with specified config and workloads.

    Args:
        config_path: Path to config file
        workloads: Tuple of workload names
        output_dir: Optional output directory override
        project: Optional Project instance for organizing results
    """
    config = load_config(config_path)
    if output_dir is not None:
        config.paths.benchmark_dir = output_dir
    cache_dir = config.paths.cache_dir / config.dataset.primary
    _, records = prepare_dataset_from_config(config.dataset, cache_dir)
    runner = BenchmarkRunner(config, project=project)
    selected_workloads = list(workloads) or DEFAULT_WORKLOADS
    queries = _default_queries(runner, records)
    filter_scenarios = runner.build_filter_scenarios(records)
    selected_filter = filter_scenarios[0].filters if filter_scenarios else None
    summaries: list[BenchmarkSummary] = []

    for repetition_index in range(config.workloads.repetitions):
        for engine_name in config.engines.enabled:
            if "a1" in selected_workloads:
                summaries.append(
                    runner.run_workload_a1(
                        engine_name,
                        records,
                        batch_size=config.workloads.ingest_batch_sizes[0],
                        repetition_index=repetition_index,
                    )
                )
            if "a2" in selected_workloads:
                summaries.append(
                    runner.run_workload_a2(
                        engine_name,
                        records,
                        query_records=queries,
                        k=config.workloads.k_values[0],
                        repetition_index=repetition_index,
                    )
                )
            if "b1" in selected_workloads and selected_filter is not None:
                summaries.append(
                    runner.run_workload_b1(
                        engine_name,
                        records,
                        filters=selected_filter,
                        query_records=queries,
                        k=config.workloads.k_values[0],
                        repetition_index=repetition_index,
                    )
                )
            if "b2" in selected_workloads:
                summaries.append(
                    runner.run_workload_b2(
                        engine_name,
                        records,
                        query_records=queries[:4],
                        batch_size=config.workloads.ingest_batch_sizes[0],
                        ingest_rate_per_second=config.workloads.ingest_rates_per_second[0],
                        k=config.workloads.k_values[0],
                        repetition_index=repetition_index,
                    )
                )
            if "b3" in selected_workloads:
                summaries.append(
                    runner.run_workload_b3(
                        engine_name,
                        records,
                        query_records=queries[:4],
                        k=config.workloads.k_values[0],
                        repetition_index=repetition_index,
                    )
                )
            if "b4" in selected_workloads:
                summaries.append(
                    runner.run_workload_b4(
                        engine_name,
                        records,
                        query_records=queries,
                        k=config.workloads.k_values[0],
                        repetition_index=repetition_index,
                    )
                )

    for summary in summaries:
        _print_summary_row(summary)


@app.command("init")
@click.option(
    "--project",
    "project_name",
    type=str,
    default="benchmark_default",
    help="Name of the project to initialize.",
    show_default=True,
)
@click.option(
    "--template",
    type=str,
    default="default",
    help="Template for initial configuration ('default' or path to config file).",
    show_default=True,
)
def init_command(project_name: str, template: str) -> None:
    """Initialize a new benchmark project with folder structure and metadata.

    Creates a new project directory with subdirectories for results, reports, and logs.
    Optionally copies a template configuration file to the project.

    Examples:
        uv run python main.py init --project my-benchmark
        uv run python main.py init --project my-benchmark --template configs/my-config.toml
    """
    if not _validate_project_name(project_name):
        console.print(
            f"[red]Error: Invalid project name '{project_name}'[/red]\n"
            "[dim]Project names must be alphanumeric (can include hyphen, underscore, or dot)[/dim]"
        )
        raise click.Abort()

    try:
        project = Project.initialize(project_name, template=template if template != "default" else "default")
        console.print(f"[green]✓ Project initialized:[/green] {project.base_path}")
        console.print(f"[dim]Config: {project.config_path}[/dim]")
        console.print(f"[dim]Results: {project.results_dir}[/dim]")
        console.print(f"[dim]Reports: {project.reports_dir}[/dim]")
        console.print(f"[dim]Logs: {project.logs_dir}[/dim]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@app.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to config file. If not provided, searches project or configs/ directory.",
)
@click.option(
    "--project",
    "project_name",
    type=str,
    default=None,
    help="Project name for organizing results. Defaults to 'benchmark_default'.",
)
@click.option("--workload", "workloads", multiple=True, help="Workload to run (multiple allowed).")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Override output directory.")
def run_command(
    config_path: Path | None,
    project_name: str | None,
    workloads: tuple[str, ...],
    output_dir: Path | None,
) -> None:
    """Run the selected workloads for the enabled engines.

    Examples:
        uv run python main.py run --project my-benchmark
        uv run python main.py run --project my-benchmark --config configs/my-config.toml
        uv run python main.py run --project my-benchmark --workload a1 --workload a2
    """
    # Get or create project
    project = _get_or_create_project(project_name)

    # Determine config path
    if config_path is None:
        config_path = _find_config(project)

    console.print(f"[dim]Project: {project.base_path}[/dim]")
    console.print(f"[dim]Config: {config_path}[/dim]")

    _execute_run(config_path, workloads, output_dir, project=project)


@app.command("run-all")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to config file. If not provided, searches project or configs/ directory.",
)
@click.option(
    "--project",
    "project_name",
    type=str,
    default=None,
    help="Project name for organizing results. Defaults to 'benchmark_default'.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Override output directory.")
def run_all_command(
    config_path: Path | None,
    project_name: str | None,
    output_dir: Path | None,
) -> None:
    """Run the default full workload suite.

    Examples:
        uv run python main.py run-all --project my-benchmark
        uv run python main.py run-all --project my-benchmark --config configs/my-config.toml
    """
    # Get or create project
    project = _get_or_create_project(project_name)

    # Determine config path
    if config_path is None:
        config_path = _find_config(project)

    console.print(f"[dim]Project: {project.base_path}[/dim]")
    console.print(f"[dim]Config: {config_path}[/dim]")

    _execute_run(config_path, tuple(DEFAULT_WORKLOADS), output_dir, project=project)


@app.command("download")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
def download_command(config_path: Path, cache_dir: Path | None) -> None:
    """Download and prepare configured dataset into local cache."""

    config = load_config(config_path)
    resolved_cache_dir = cache_dir or (config.paths.cache_dir / config.dataset.primary)
    manifest, records = download_dataset_from_config(config.dataset, resolved_cache_dir)
    console.print(f"dataset={manifest.dataset_name}")
    console.print(f"variant={manifest.dataset_variant}")
    console.print(f"records={len(records)}")
    console.print(f"cache={resolved_cache_dir}")


@app.command("report")
@click.option(
    "--project",
    "project_name",
    type=str,
    default=None,
    help="Project name for organizing results. Defaults to 'benchmark_default'.",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Results directory. If not provided, uses project results directory.",
)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def report_command(
    project_name: str | None,
    input_dir: Path | None,
    output_path: Path,
) -> None:
    """Consolidate all run summaries into one analytics artifact.

    Examples:
        uv run python main.py report --project my-benchmark --output results.parquet
        uv run python main.py report --input .benchmarks/results --output results.parquet
    """
    # Use project results dir if not provided
    if input_dir is None:
        project = _get_or_create_project(project_name)
        input_dir = project.results_dir
        console.print(f"[dim]Project: {project.base_path}[/dim]")

    report_path = write_consolidated_report(input_dir, output_path)
    index_path = write_results_index(input_dir, input_dir / "index.parquet")
    console.print(f"report={report_path}")
    console.print(f"index={index_path}")


@app.command("inspect")
@click.option("--run-id", required=True, help="Run ID to inspect.")
@click.option(
    "--project",
    "project_name",
    type=str,
    default=None,
    help="Project name. Defaults to 'benchmark_default'.",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Results directory. If not provided, uses project results directory.",
)
@click.option("--type", "record_type", type=str, default=None, help="Record type to display (queries or ingest).")
@click.option("--summary", is_flag=True, default=False, help="Show summary only.")
def inspect_command(
    run_id: str,
    project_name: str | None,
    input_dir: Path | None,
    record_type: str | None,
    summary: bool,
) -> None:
    """Inspect one run summary or its raw query/ingest records.

    Examples:
        uv run python main.py inspect --project my-benchmark --run-id run-123
        uv run python main.py inspect --run-id run-123 --type queries
    """
    # Use project results dir if not provided
    if input_dir is None:
        project = _get_or_create_project(project_name)
        input_dir = project.results_dir
        console.print(f"[dim]Project: {project.base_path}[/dim]")

    payload = inspect_run(input_dir, run_id)
    run_summary = payload["summary"]
    raw_records = payload["raw_records"]

    if summary or record_type is None:
        table = Table(title=f"Run {run_summary.run_id}")
        table.add_column("Field")
        table.add_column("Value")
        for field in [
            "engine",
            "workload",
            "dataset_name",
            "mean_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "recall_at_k",
            "status",
        ]:
            table.add_row(field, str(getattr(run_summary, field)))
        console.print(table)
        if record_type is None:
            return

    phase_name = "query" if record_type == "queries" else "ingest"
    filtered_records = [record for record in raw_records if record.get("phase") == phase_name]
    preview = filtered_records[:20]

    table = Table(title=f"{run_summary.run_id} {phase_name}")
    columns = sorted({key for record in preview for key in record})
    for column in columns:
        table.add_column(column)
    for record in preview:
        table.add_row(*[str(record.get(column, "")) for column in columns])
    console.print(table)
    console.print(f"records={len(filtered_records)}")
    console.print(f"schema={sorted({key for record in filtered_records for key in record})}")


@app.command("charts")
@click.option(
    "--project",
    "project_name",
    type=str,
    default=None,
    help="Project name. Defaults to 'benchmark_default'.",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing run results. If not provided, uses project results directory.",
    show_default=False,
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory where PNG chart files will be written. If not provided, uses project charts directory.",
    show_default=False,
)
def charts_command(
    project_name: str | None,
    input_dir: Path | None,
    output_dir: Path | None,
) -> None:
    """Generate 6 comparative benchmark charts as PNG files.

    Charts produced:

      chart_01_latency.png       — mean / p95 / p99 latency per engine × workload

      chart_02_qps.png           — QPS / throughput per engine × workload

      chart_03_recall.png        — Recall@K per engine × workload

      chart_04_peak_ram.png      — Peak RAM (MB) per engine × workload

      chart_05_disk_footprint.png — Disk footprint (MB) per engine

      chart_06_build_time.png    — Build time / Time-to-serve per engine

    Examples:
        uv run python main.py charts --project my-benchmark
        uv run python main.py charts --project my-benchmark --output charts/
    """
    # Use project directories if not provided
    if input_dir is None or output_dir is None:
        project = _get_or_create_project(project_name)
        input_dir = input_dir or project.results_dir
        output_dir = output_dir or project.charts_dir
        console.print(f"[dim]Project: {project.base_path}[/dim]")

    paths = generate_charts(input_dir, output_dir)
    if not paths:
        console.print("[yellow]No benchmark results found. Run benchmarks first.[/yellow]")
        return
    for path in paths:
        console.print(f"chart={path}")
    console.print(f"[green]Generated {len(paths)} charts in {output_dir}[/green]")


@app.command("analyze")
@click.option(
    "--project",
    "project_name",
    type=str,
    default=None,
    help="Project name. Defaults to benchmark_default.",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom results directory. If omitted, uses project results_dir.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path (.md, .html, or .json)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "html", "json"]),
    default="markdown",
    help="Output format.",
    show_default=True,
)
@click.option(
    "--report-type",
    "report_type",
    type=click.Choice(["executive_summary", "detailed", "statistical_only"]),
    default="detailed",
    help="Report type.",
    show_default=True,
)
@click.option(
    "--include-charts",
    is_flag=True,
    default=True,
    help="Include charts reference in report.",
    show_default=True,
)
@click.option(
    "--dataset-name",
    type=str,
    default=None,
    help="Dataset name (auto-detect if omitted).",
)
def analyze_command(
    project_name: str | None,
    input_dir: Path | None,
    output_path: Path,
    output_format: str,
    report_type: str,
    include_charts: bool,
    dataset_name: str | None,
) -> None:
    """Generate comprehensive analysis from benchmark results.

    Creates a structured report analyzing benchmark run summaries from a project.
    Supports multiple output formats (Markdown, HTML, JSON) and report types
    (executive summary, detailed, statistical analysis only).

    Examples:
        uv run python main.py analyze --project my-benchmark --output report.md
        uv run python main.py analyze --project my-benchmark --output report.html --format html
        uv run python main.py analyze --project my-benchmark --output report.json --format json --report-type executive_summary
    """
    try:
        logger.info(
            f"Starting analysis: project={project_name}, format={output_format}, "
            f"report_type={report_type}, include_charts={include_charts}"
        )

        # Determine results directory
        if input_dir is None:
            project = _get_or_create_project(project_name)
            input_dir = project.results_dir
            console.print(f"[dim]Project: {project.base_path}[/dim]")
        else:
            input_dir = input_dir.resolve()

        logger.info(f"Using results directory: {input_dir}")

        # Validate that results directory exists
        if not input_dir.exists():
            console.print(f"[red]Error: Results directory not found: {input_dir}[/red]")
            logger.error(f"Results directory does not exist: {input_dir}")
            raise click.Abort()

        # Generate report
        from benchvec.generate_analysis import AnalysisConfig, OutputFormat, ReportType, generate_report
        
        config = AnalysisConfig(
            input_dir=input_dir.parent if (input_dir / "results").exists() else input_dir,
            output_path=output_path.resolve(),
            output_format=OutputFormat(output_format),
            report_type=ReportType(report_type),
            include_charts=include_charts,
            dataset_name=dataset_name,
            suppress_sections=[],
        )
        
        console.print(f"[dim]Generating report from: {input_dir}[/dim]")
        result_path = generate_report(config)
        logger.info(f"Report written to: {result_path}")

        # Print success message
        console.print(f"[green]✓ Analysis complete:[/green] {result_path}")
        console.print(f"[dim]Format: {output_format}[/dim]")
        console.print(f"[dim]Report type: {report_type}[/dim]")
        console.print(f"[dim]Size: {result_path.stat().st_size} bytes[/dim]")

    except FileNotFoundError as e:
        logger.error(f"File not found during analysis: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except ValueError as e:
        logger.error(f"Invalid value during analysis: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        console.print(f"[red]Error during analysis: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    app()
