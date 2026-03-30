"""
Simplified test with 5K synthetic Cohere data

This test generates 5,000 synthetic records based on the Cohere dataset schema
and runs a simplified analysis pipeline.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchvec.generate_analysis import (
    AnalysisConfig,
    OutputFormat,
    ReportType,
    generate_report,
)


def generate_synthetic_cohere_data(n_records: int = 5000) -> pd.DataFrame:
    """Generate synthetic Cohere benchmark data.
    
    Args:
        n_records: Number of records to generate
        
    Returns:
        DataFrame with synthetic benchmark results
    """
    engines = ["chromadb", "faiss", "lancedb"]
    workloads = ["a1", "a2", "b1", "b2", "b3", "b4"]
    
    np.random.seed(42)
    
    data = {
        "engine": np.random.choice(engines, n_records),
        "workload": np.random.choice(workloads, n_records),
        "run_id": [f"run-{i:06d}" for i in range(n_records)],
        "mean_latency_ms": np.random.gamma(2, 10, n_records),  # ~20ms avg
        "p99_latency_ms": np.random.gamma(2.5, 20, n_records),  # ~50ms
        "qps": np.random.gamma(3, 30, n_records),  # ~100 QPS
        "recall_at_k": np.random.beta(10, 1, n_records) * 0.99 + 0.01,  # 0.01-1.0
        "peak_ram_mb": np.random.gamma(2, 100, n_records),  # ~200MB
        "dataset_name": "cohere-wikipedia-22-12",
        "timestamp": pd.date_range("2026-01-01", periods=n_records, freq="1min"),
    }
    
    df = pd.DataFrame(data)
    return df


def test_simplified_cohere_analysis():
    """Test simplified analysis pipeline with 5K Cohere records."""
    
    print("=" * 70)
    print("SIMPLIFIED COHERE ANALYSIS TEST - 5K Records")
    print("=" * 70)
    
    # 1. Generate synthetic data
    print("\n1. Generating 5,000 synthetic Cohere records...")
    df = generate_synthetic_cohere_data(5000)
    print(f"   ✓ Generated {len(df)} records")
    print(f"   ✓ Engines: {df['engine'].unique().tolist()}")
    print(f"   ✓ Workloads: {df['workload'].unique().tolist()}")
    
    # 2. Create temporary directory with results
    print("\n2. Setting up temporary results directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create reports directory and save consolidated parquet
        reports_dir = tmpdir / "reports"
        reports_dir.mkdir(parents=True)
        df.to_parquet(reports_dir / "final.parquet")
        print(f"   ✓ Saved to {reports_dir / 'final.parquet'}")
        
        # 3. Test all format/type combinations
        print("\n3. Testing analysis generation (3 formats × 3 types = 9 combinations)...")
        formats = [OutputFormat.markdown, OutputFormat.json, OutputFormat.html]
        types = [ReportType.executive_summary, ReportType.detailed, ReportType.statistical_only]
        
        results = []
        for fmt in formats:
            for rtype in types:
                output_file = tmpdir / f"report_{fmt.value}_{rtype.value}.{fmt.value}"
                
                config = AnalysisConfig(
                    input_dir=tmpdir,
                    output_path=output_file,
                    output_format=fmt,
                    report_type=rtype,
                    dataset_name="Cohere Wikipedia 22-12",
                    include_charts=False,  # Simplified - no charts
                )
                
                try:
                    result_path = generate_report(config)
                    size = result_path.stat().st_size
                    results.append((fmt.value, rtype.value, size, "✓"))
                    print(f"   ✓ {fmt.value:10s} + {rtype.value:20s} = {size:6d} bytes")
                except Exception as e:
                    results.append((fmt.value, rtype.value, 0, f"✗ {str(e)[:30]}"))
                    print(f"   ✗ {fmt.value:10s} + {rtype.value:20s} = ERROR: {e}")
        
        # 4. Verify all succeeded
        print("\n4. Verification Summary:")
        successful = sum(1 for r in results if r[3] == "✓")
        total = len(results)
        print(f"   {successful}/{total} combinations successful")
        
        if successful == total:
            print("\n✅ ALL TESTS PASSED")
            return True
        else:
            print("\n❌ SOME TESTS FAILED")
            return False


def test_statistical_analysis():
    """Test statistical analysis on synthetic data."""
    
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    df = generate_synthetic_cohere_data(5000)
    
    # By engine
    print("\nMetrics by Engine:")
    by_engine = df.groupby("engine").agg({
        "mean_latency_ms": "mean",
        "qps": "mean",
        "recall_at_k": "mean",
        "peak_ram_mb": "mean",
    }).round(2)
    print(by_engine)
    
    # By workload
    print("\nMetrics by Workload:")
    by_workload = df.groupby("workload").agg({
        "mean_latency_ms": "mean",
        "qps": "mean",
        "recall_at_k": "mean",
        "peak_ram_mb": "mean",
    }).round(2)
    print(by_workload)
    
    # Overall stats
    print("\nOverall Statistics:")
    print(f"  Total records:       {len(df)}")
    print(f"  Avg latency:         {df['mean_latency_ms'].mean():.2f}ms")
    print(f"  Avg QPS:             {df['qps'].mean():.2f}")
    print(f"  Avg recall:          {df['recall_at_k'].mean():.4f}")
    print(f"  Avg peak RAM:        {df['peak_ram_mb'].mean():.2f}MB")


if __name__ == "__main__":
    test_simplified_cohere_analysis()
    test_statistical_analysis()
