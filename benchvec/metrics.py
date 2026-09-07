"""Benchmark metric helpers and summary persistence."""

from __future__ import annotations

import datetime as dt
import json
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True, frozen=True)
class LatencyStats:
    """Latency aggregates used by workload summaries."""

    mean_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float


class BenchmarkSummary(BaseModel):
    """Aggregated workload output schema aligned with PRD03."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: str
    engine: str
    engine_version: str
    dataset_name: str
    dataset_group: str
    dataset_size: int = Field(ge=0)
    dimension: int = Field(ge=0)
    index_type: str
    distance_metric: str
    workload: str
    filter_selectivity: float | None = Field(default=None, ge=0, le=1)
    k: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    index_params: dict[str, Any] = Field(default_factory=dict)
    search_params: dict[str, Any] = Field(default_factory=dict)
    n_warmup_queries: int = Field(ge=0)
    n_measured_queries: int = Field(ge=0)
    n_repetitions: int = Field(ge=1)
    load_time_s: float = Field(ge=0)
    build_time_s: float = Field(ge=0)
    time_to_serve_s: float = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    p99_latency_ms: float = Field(ge=0)
    std_latency_ms: float = Field(ge=0)
    qps: float = Field(ge=0)
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    peak_ram_mb: float = Field(ge=0)
    disk_size_mb: float = Field(ge=0)
    reopen_time_s: float = Field(ge=0)
    timeout_policy_s: int = Field(ge=0)
    status: str
    error_message: str | None = None
    host_fingerprint: dict[str, Any]
    raw_data_path: str
    concurrency: int | None = Field(default=None, ge=1)
    delete_time_s: float | None = Field(default=None, ge=0)
    upsert_time_s: float | None = Field(default=None, ge=0)
    post_delete_recall: float | None = Field(default=None, ge=0, le=1)


def new_run_id() -> str:
    """Generate a stable run identifier for one workload execution."""

    return f"run-{uuid.uuid4().hex[:12]}"


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def process_ram_mb() -> float:
    """Read the current resident memory of the process in MB."""

    process = psutil.Process()
    return float(process.memory_info().rss / (1024**2))


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Compute a percentile using linear interpolation."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile_value
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def summarize_latencies(latencies_ms: Sequence[float]) -> LatencyStats:
    """Aggregate latency measurements for summary reporting."""

    if not latencies_ms:
        return LatencyStats(mean_ms=0.0, p95_ms=0.0, p99_ms=0.0, std_ms=0.0)
    values = [float(latency) for latency in latencies_ms]
    return LatencyStats(
        mean_ms=float(statistics.fmean(values)),
        p95_ms=percentile(values, 0.95),
        p99_ms=percentile(values, 0.99),
        std_ms=float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    )


def calculate_recall_at_k(
    returned_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Compute Recall@K from returned IDs and exact neighbors."""

    if k <= 0:
        raise ValueError("k must be positive")
    expected = set(ground_truth_ids[:k])
    if not expected:
        return 0.0
    returned = set(returned_ids[:k])
    return float(len(expected & returned) / len(expected))


def queries_per_second(query_count: int, elapsed_seconds: float) -> float:
    """Compute sustained query throughput."""

    if elapsed_seconds <= 0:
        return 0.0
    return float(query_count / elapsed_seconds)


def write_summary(output_dir: str | Path, summary: BenchmarkSummary) -> Path:
    """Persist one aggregated summary as Parquet."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(mode="json")
    payload["index_params"] = json.dumps(payload["index_params"], sort_keys=True)
    payload["search_params"] = json.dumps(payload["search_params"], sort_keys=True)
    payload["host_fingerprint"] = json.dumps(payload["host_fingerprint"], sort_keys=True)
    table = pa.Table.from_pylist([payload])
    target_path = target_dir / "summary.parquet"
    pq.write_table(table, target_path)
    return target_path


def load_summary(summary_path: str | Path) -> BenchmarkSummary:
    """Load one persisted benchmark summary."""

    table = pq.read_table(summary_path)
    rows = cast(list[dict[str, Any]], table.to_pylist())
    payload = rows[0]
    payload["index_params"] = json.loads(payload["index_params"])
    payload["search_params"] = json.loads(payload["search_params"])
    payload["host_fingerprint"] = json.loads(payload["host_fingerprint"])
    return BenchmarkSummary.model_validate(payload)


def write_env_snapshot(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Persist the execution environment snapshot as JSON."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "env.json"
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target_path
