"""Tests for benchvec.generate_analysis module."""
import json
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from benchvec.generate_analysis import (
    AnalysisConfig,
    OutputFormat,
    ReportType,
    detect_dataset_name,
    generate_report,
    load_results,
)


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_config_creation_minimal(self, tmp_path: Path) -> None:
        """Test creating AnalysisConfig with minimal arguments."""
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=tmp_path / "report.md",
        )
        assert config.input_dir == tmp_path
        assert config.output_path == tmp_path / "report.md"
        assert config.output_format == OutputFormat.markdown
        assert config.report_type == ReportType.detailed

    def test_config_creation_full(self, tmp_path: Path) -> None:
        """Test creating AnalysisConfig with all arguments."""
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=tmp_path / "report.json",
            output_format=OutputFormat.json,
            report_type=ReportType.executive_summary,
            include_charts=False,
            dataset_name="Custom Dataset",
            suppress_sections=["charts", "methodology"],
        )
        assert config.input_dir == tmp_path
        assert config.output_path == tmp_path / "report.json"
        assert config.output_format == OutputFormat.json
        assert config.report_type == ReportType.executive_summary
        assert config.include_charts is False
        assert config.dataset_name == "Custom Dataset"
        assert config.suppress_sections == ["charts", "methodology"]

    def test_config_output_format_enum(self, tmp_path: Path) -> None:
        """Test that output_format accepts valid enum values."""
        for fmt in OutputFormat:
            config = AnalysisConfig(
                input_dir=tmp_path,
                output_path=tmp_path / "report.md",
                output_format=fmt,
            )
            assert config.output_format == fmt

    def test_config_report_type_enum(self, tmp_path: Path) -> None:
        """Test that report_type accepts valid enum values."""
        for rtype in ReportType:
            config = AnalysisConfig(
                input_dir=tmp_path,
                output_path=tmp_path / "report.md",
                report_type=rtype,
            )
            assert config.report_type == rtype


class TestDetectDatasetName:
    """Tests for detect_dataset_name function."""

    def test_detect_cohere_from_dataframe(self) -> None:
        """Test detecting Cohere dataset from dataframe column."""
        df = pd.DataFrame({
            "dataset_name": ["cohere-wikipedia-22-12"],
            "metric": [100],
        })
        name = detect_dataset_name(df, Path("/some/cohere-2026"))
        assert "Cohere" in name or "wikipedia" in name.lower()

    def test_detect_sift_from_dataframe(self) -> None:
        """Test detecting SIFT dataset from dataframe column."""
        df = pd.DataFrame({
            "dataset_name": ["sift1m"],
            "metric": [100],
        })
        name = detect_dataset_name(df, Path("/some/path"))
        assert "SIFT" in name

    def test_detect_gist_from_dataframe(self) -> None:
        """Test detecting GIST dataset from dataframe column."""
        df = pd.DataFrame({
            "dataset_name": ["gist1m"],
            "metric": [100],
        })
        name = detect_dataset_name(df, Path("/some/path"))
        assert "GIST" in name

    def test_detect_from_project_name_cohere(self) -> None:
        """Test detecting dataset from project directory name (Cohere)."""
        df = pd.DataFrame({"metric": [100]})
        name = detect_dataset_name(df, Path("/some/cohere-benchmark"))
        assert "Cohere" in name

    def test_detect_from_project_name_sift(self) -> None:
        """Test detecting dataset from project directory name (SIFT)."""
        df = pd.DataFrame({"metric": [100]})
        name = detect_dataset_name(df, Path("/some/sift1m-benchmark"))
        assert "SIFT" in name


class TestLoadResults:
    """Tests for load_results function."""

    def test_load_results_with_final_parquet(self, tmp_path: Path) -> None:
        """Test loading consolidated final.parquet file."""
        # Create test data
        test_df = pd.DataFrame({
            "engine": ["chromadb", "faiss"],
            "workload": ["a1", "a2"],
            "latency_mean": [10.5, 20.3],
            "qps": [100.0, 50.0],
        })
        
        # Create reports directory and final.parquet
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        test_df.to_parquet(reports_dir / "final.parquet")
        
        # Load and verify
        result = load_results(tmp_path)
        assert len(result) == 2
        assert "engine" in result.columns
        assert result["engine"].tolist() == ["chromadb", "faiss"]

    def test_load_results_with_individual_summaries(self, tmp_path: Path) -> None:
        """Test loading individual summary.parquet files."""
        # Create test data
        test_df = pd.DataFrame({
            "engine": ["chromadb"],
            "workload": ["a1"],
            "latency_mean": [10.5],
            "qps": [100.0],
        })
        
        # Create results directory structure: results/engine/workload/run-id/summary.parquet
        results_dir = tmp_path / "results"
        run_dir = results_dir / "chromadb" / "a1" / "run-12345"
        run_dir.mkdir(parents=True)
        test_df.to_parquet(run_dir / "summary.parquet")
        
        # Load and verify
        result = load_results(results_dir)
        assert len(result) == 1
        assert "engine" in result.columns
        assert result["engine"].iloc[0] == "chromadb"

    def test_load_results_no_data_raises_error(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when no results exist."""
        with pytest.raises(FileNotFoundError, match="No results found"):
            load_results(tmp_path)


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_report_markdown_format(self, tmp_path: Path) -> None:
        """Test generating report in markdown format."""
        # Create test data
        test_df = pd.DataFrame({
            "engine": ["chromadb", "faiss"],
            "workload": ["a1", "a2"],
            "latency_mean": [10.5, 20.3],
            "qps": [100.0, 50.0],
            "recall": [0.99, 0.95],
            "mean_latency_ms": [10.5, 20.3],
            "recall_at_k": [0.99, 0.95],
        })
        
        # Create reports directory
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        test_df.to_parquet(reports_dir / "final.parquet")
        
        # Create config
        output_file = tmp_path / "report.md"
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=output_file,
            output_format=OutputFormat.markdown,
        )
        
        # Generate report
        result_path = generate_report(config)
        
        # Verify
        assert result_path == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_report_json_format(self, tmp_path: Path) -> None:
        """Test generating report in JSON format."""
        # Create test data
        test_df = pd.DataFrame({
            "engine": ["chromadb"],
            "workload": ["a1"],
            "latency_mean": [10.5],
            "qps": [100.0],
            "recall": [0.99],
        })
        
        # Create reports directory
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        test_df.to_parquet(reports_dir / "final.parquet")
        
        # Create config
        output_file = tmp_path / "report.json"
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=output_file,
            output_format=OutputFormat.json,
        )
        
        # Generate report
        result_path = generate_report(config)
        
        # Verify
        assert result_path == output_file
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        assert "dataset" in data
        assert "report_type" in data
        assert "sections" in data

    def test_generate_report_executive_summary_type(self, tmp_path: Path) -> None:
        """Test generating report with executive_summary type."""
        # Create test data
        test_df = pd.DataFrame({
            "engine": ["chromadb"],
            "workload": ["a1"],
            "latency_mean": [10.5],
            "qps": [100.0],
            "recall": [0.99],
        })
        
        # Create reports directory
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        test_df.to_parquet(reports_dir / "final.parquet")
        
        # Create config
        output_file = tmp_path / "report.md"
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=output_file,
            report_type=ReportType.executive_summary,
        )
        
        # Generate report
        result_path = generate_report(config)
        
        # Verify
        assert result_path == output_file
        assert output_file.exists()

    def test_generate_report_custom_dataset_name(self, tmp_path: Path) -> None:
        """Test generating report with custom dataset name."""
        # Create test data
        test_df = pd.DataFrame({
            "engine": ["chromadb"],
            "workload": ["a1"],
            "latency_mean": [10.5],
            "qps": [100.0],
            "recall": [0.99],
        })
        
        # Create reports directory
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        test_df.to_parquet(reports_dir / "final.parquet")
        
        # Create config
        output_file = tmp_path / "report.md"
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=output_file,
            dataset_name="MyCustomDataset",
        )
        
        # Generate report
        result_path = generate_report(config)
        
        # Verify
        assert result_path == output_file
        content = output_file.read_text()
        assert "MyCustomDataset" in content


class TestOutputFormats:
    """Tests for different output formats."""

    def test_output_format_enum_values(self) -> None:
        """Test that OutputFormat enum has expected values."""
        assert OutputFormat.markdown.value == "markdown"
        assert OutputFormat.html.value == "html"
        assert OutputFormat.json.value == "json"

    def test_report_type_enum_values(self) -> None:
        """Test that ReportType enum has expected values."""
        assert ReportType.executive_summary.value == "executive_summary"
        assert ReportType.detailed.value == "detailed"
        assert ReportType.statistical_only.value == "statistical_only"


class TestSignificanceSection:
    """Tests for the Holm-corrected pairwise significance section."""

    def test_markdown_includes_significance_when_parquet_present(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from benchvec.statistics import STATS_PAIRWISE_SCHEMA

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        pd.DataFrame({"engine": ["faiss"], "workload": ["a2"], "mean_latency_ms": [1.0], "qps": [100.0], "recall_at_k": [0.99]}).to_parquet(
            reports_dir / "final.parquet"
        )
        pairwise_rows = [
            {
                "workload": "a2",
                "k": 10,
                "concurrency_level": -1,
                "engine_a": "faiss",
                "engine_b": "chromadb",
                "delta_mean_pct": 42.0,
                "u_statistic": 12.0,
                "p_value": 0.003,
                "holm_adjusted_p": 0.009,
                "significant_at_0_05": True,
            }
        ]
        pq.write_table(
            pa.Table.from_pylist(pairwise_rows, schema=STATS_PAIRWISE_SCHEMA),
            tmp_path / "stats_pairwise.parquet",
        )

        output_file = tmp_path / "report.md"
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=output_file,
            output_format=OutputFormat.markdown,
        )
        result_path = generate_report(config)

        assert result_path == output_file
        content = output_file.read_text(encoding="utf-8")
        assert "Statistical Significance" in content
        assert "Holm" in content
        assert "+42.0%" in content

    def test_markdown_omits_significance_when_parquet_absent(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        pd.DataFrame({"engine": ["faiss"], "workload": ["a2"], "mean_latency_ms": [1.0], "qps": [100.0], "recall_at_k": [0.99]}).to_parquet(
            reports_dir / "final.parquet"
        )

        output_file = tmp_path / "report.md"
        config = AnalysisConfig(
            input_dir=tmp_path,
            output_path=output_file,
            output_format=OutputFormat.markdown,
        )
        generate_report(config)

        content = output_file.read_text(encoding="utf-8")
        assert "Statistical Significance" not in content
