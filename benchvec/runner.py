"""Workload runner for reproducible BenchVec benchmark execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from benchvec.config import BenchmarkConfig, EngineName
from benchvec.datasets import PreparedRecord, stable_hash
from benchvec.engines import (
    BenchmarkEngine,
    ChromaEngine,
    EngineRecord,
    FaissEngine,
    FilterClause,
    LanceDbEngine,
    MetadataFilter,
)
from benchvec.ground_truth import compute_exact_ground_truth
from benchvec.host import HostFingerprint, collect_host_fingerprint
from benchvec.metrics import (
    BenchmarkSummary,
    calculate_recall_at_k,
    new_run_id,
    process_ram_mb,
    queries_per_second,
    summarize_latencies,
    utc_timestamp,
    write_env_snapshot,
    write_summary,
)
from benchvec.project import Project
from benchvec.raw_store import RawRecord, write_raw_records
from benchvec.timeouts import TimeoutStatus, execute_with_timeout

logger = logging.getLogger(__name__)

EngineFactory = Callable[[EngineName, Path], BenchmarkEngine]


@dataclass(slots=True, frozen=True)
class FilterScenario:
    """One filtered-search scenario with observed selectivity."""

    name: str
    filters: MetadataFilter
    observed_selectivity: float


@dataclass(slots=True)
class RunArtifacts:
    """Filesystem targets for one benchmark execution."""

    run_id: str
    run_dir: Path
    raw_dir: Path
    engine_dir: Path
    log_path: Path


class BenchmarkRunner:
    """Orchestrates benchmark workloads on top of the unified engine contract."""

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        project: Project | None = None,
        host_fingerprint: HostFingerprint | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.config = config
        self.project = project
        self.host_fingerprint = host_fingerprint or collect_host_fingerprint()
        self.engine_factory = engine_factory or self._build_engine

    def default_query_records(
        self,
        records: Sequence[PreparedRecord],
        *,
        query_count: int | None = None,
    ) -> list[PreparedRecord]:
        """Select deterministic query records from the prepared dataset."""

        target_count = query_count or min(
            self.config.dataset.ground_truth_query_count,
            len(records),
        )
        ranked_records = sorted(
            records,
            key=lambda record: stable_hash(f"{self.config.seed}:{record.id}:query"),
        )
        return ranked_records[:target_count]

    def build_filter_scenarios(self, records: Sequence[PreparedRecord]) -> list[FilterScenario]:
        """Derive deterministic filter scenarios near PRD03 selectivity targets."""

        candidates: list[tuple[str, MetadataFilter, float]] = []
        total_records = max(len(records), 1)

        def observed(filters: MetadataFilter) -> float:
            matched = sum(1 for record in records if filters.matches(record_to_metadata(record)))
            return float(matched / total_records)

        unique_tags = sorted({record.tag for record in records})
        for tag in unique_tags:
            filters = MetadataFilter((FilterClause("tag", "eq", tag),))
            candidates.append((f"tag:{tag}", filters, observed(filters)))

        unique_categories = sorted({record.categoria for record in records})
        for category in unique_categories:
            filters = MetadataFilter((FilterClause("categoria", "eq", category),))
            candidates.append((f"categoria:{category}", filters, observed(filters)))

        unique_authors = sorted({record.autor for record in records})
        for author in unique_authors:
            filters = MetadataFilter((FilterClause("autor", "eq", author),))
            candidates.append((f"autor:{author}", filters, observed(filters)))

        unique_dates = sorted({record.data_criacao for record in records})
        date_step = max(len(unique_dates) // 8, 1)
        for date_value in unique_dates[::date_step]:
            filters = MetadataFilter((FilterClause("data_criacao", "gte", date_value),))
            candidates.append((f"date>={date_value}", filters, observed(filters)))

        for author in unique_authors:
            for category in unique_categories:
                filters = MetadataFilter(
                    (
                        FilterClause("autor", "eq", author),
                        FilterClause("categoria", "eq", category),
                    )
                )
                selectivity = observed(filters)
                if selectivity > 0:
                    candidates.append((f"{author}:{category}", filters, selectivity))

        targets = {
            "low": 0.5,
            "medium": 0.1,
            "high": 0.01,
            "extreme": 0.001,
        }
        scenarios: list[FilterScenario] = []
        for name, target in targets.items():
            best_name, best_filter, best_selectivity = min(
                candidates,
                key=lambda candidate: (abs(candidate[2] - target), candidate[0]),
            )
            scenarios.append(
                FilterScenario(
                    name=f"{name}:{best_name}",
                    filters=best_filter,
                    observed_selectivity=best_selectivity,
                )
            )
        return scenarios

    def run_workload_a1(
        self,
        engine_name: EngineName,
        records: Sequence[PreparedRecord],
        *,
        batch_size: int,
        repetition_index: int = 0,
    ) -> BenchmarkSummary:
        """Execute workload A1 bulk ingestion."""

        artifacts = self._prepare_artifacts(engine_name, workload="a1")
        engine = self.engine_factory(engine_name, artifacts.engine_dir)
        raw_records: list[RawRecord] = []
        peak_ram_mb = process_ram_mb()

        def ingest_phase() -> tuple[float, list[float]]:
            engine.reset()
            engine.connect()
            latencies_ms: list[float] = []
            cumulative_vectors = 0
            for batch_index, batch in enumerate(chunk_records(records, batch_size), start=0):
                engine_batch = [record_to_engine_record(record) for record in batch]
                started_at = time.perf_counter()
                engine.insert(engine_batch)
                elapsed_seconds = time.perf_counter() - started_at
                latency_ms = elapsed_seconds * 1000
                cumulative_vectors += len(batch)
                ram_after_mb = process_ram_mb()
                latencies_ms.append(latency_ms)
                raw_records.append(
                    RawRecord(
                        run_id=artifacts.run_id,
                        engine=engine_name,
                        workload="a1",
                        phase="ingest",
                        status="success",
                        timestamp_utc=utc_timestamp(),
                        repetition_index=repetition_index,
                        batch_index=batch_index,
                        batch_size=len(batch),
                        cumulative_vectors=cumulative_vectors,
                        batch_latency_ms=latency_ms,
                        throughput_vps=(len(batch) / elapsed_seconds if elapsed_seconds else 0.0),
                        ram_mb=ram_after_mb,
                        ram_after_mb=ram_after_mb,
                    )
                )
            started_at = time.perf_counter()
            engine.build_index()
            build_time_s = time.perf_counter() - started_at
            return build_time_s, latencies_ms

        ingest_result = execute_with_timeout(
            ingest_phase,
            self.config.timeouts.ingest_batch_seconds,
        )
        build_time_s = 0.0
        batch_latencies_ms: list[float] = []
        if ingest_result.payload is not None:
            build_time_s, batch_latencies_ms = ingest_result.payload
        peak_ram_mb = max(peak_ram_mb, self._peak_ram_from_raw(raw_records))
        engine.persist()
        write_raw_records(artifacts.raw_dir, raw_records)
        summary = self._build_summary(
            artifacts=artifacts,
            engine=engine,
            records=records,
            workload="a1",
            latencies_ms=batch_latencies_ms,
            load_time_s=sum(batch_latencies_ms) / 1000,
            build_time_s=build_time_s,
            k=None,
            batch_size=batch_size,
            recall_at_k=None,
            peak_ram_mb=peak_ram_mb,
            reopen_time_s=0.0,
            timeout_policy_s=self.config.timeouts.ingest_batch_seconds,
            status=ingest_result.status.value,
            error_message=ingest_result.error_message,
            n_measured_queries=len(batch_latencies_ms),
        )
        self._persist_run_metadata(artifacts, summary)
        return summary

    def run_workload_a2(
        self,
        engine_name: EngineName,
        records: Sequence[PreparedRecord],
        *,
        query_records: Sequence[PreparedRecord] | None = None,
        k: int,
        repetition_index: int = 0,
    ) -> BenchmarkSummary:
        """Execute workload A2 pure vector search with exact recall."""

        queries = list(query_records or self.default_query_records(records))
        artifacts = self._prepare_artifacts(engine_name, workload="a2")
        engine = self.engine_factory(engine_name, artifacts.engine_dir)
        raw_records: list[RawRecord] = []
        load_time_s, build_time_s = self._load_engine(engine, records)
        ground_truth = self._ground_truth_for_queries(records, queries, k=k)
        measurement = self._run_query_phase(
            engine=engine,
            engine_name=engine_name,
            workload="a2",
            run_id=artifacts.run_id,
            queries=queries,
            raw_records=raw_records,
            ground_truth=ground_truth,
            k=k,
            filters=None,
            repetition_index=repetition_index,
        )
        write_raw_records(artifacts.raw_dir, raw_records)
        summary = self._build_summary(
            artifacts=artifacts,
            engine=engine,
            records=records,
            workload="a2",
            latencies_ms=measurement.latencies_ms,
            load_time_s=load_time_s,
            build_time_s=build_time_s,
            k=k,
            batch_size=None,
            recall_at_k=measurement.mean_recall,
            peak_ram_mb=max(process_ram_mb(), self._peak_ram_from_raw(raw_records)),
            reopen_time_s=0.0,
            timeout_policy_s=self.config.timeouts.measured_query_seconds,
            status=measurement.status,
            error_message=measurement.error_message,
            n_measured_queries=len(measurement.latencies_ms),
        )
        self._persist_run_metadata(artifacts, summary)
        return summary

    def run_workload_b1(
        self,
        engine_name: EngineName,
        records: Sequence[PreparedRecord],
        *,
        filters: MetadataFilter,
        query_records: Sequence[PreparedRecord] | None = None,
        k: int,
        repetition_index: int = 0,
    ) -> BenchmarkSummary:
        """Execute workload B1 filtered vector search."""

        queries = list(query_records or self.default_query_records(records))
        artifacts = self._prepare_artifacts(engine_name, workload="b1")
        engine = self.engine_factory(engine_name, artifacts.engine_dir)
        raw_records: list[RawRecord] = []
        load_time_s, build_time_s = self._load_engine(engine, records)
        filtered_records = [
            record for record in records if filters.matches(record_to_metadata(record))
        ]
        ground_truth = self._ground_truth_for_queries(filtered_records, queries, k=k)
        measurement = self._run_query_phase(
            engine=engine,
            engine_name=engine_name,
            workload="b1",
            run_id=artifacts.run_id,
            queries=queries,
            selectivity_records=records,
            raw_records=raw_records,
            ground_truth=ground_truth,
            k=k,
            filters=filters,
            repetition_index=repetition_index,
        )
        write_raw_records(artifacts.raw_dir, raw_records)
        summary = self._build_summary(
            artifacts=artifacts,
            engine=engine,
            records=records,
            workload="b1",
            latencies_ms=measurement.latencies_ms,
            load_time_s=load_time_s,
            build_time_s=build_time_s,
            k=k,
            batch_size=None,
            filter_selectivity=(len(filtered_records) / len(records)) if records else 0.0,
            recall_at_k=measurement.mean_recall,
            peak_ram_mb=max(process_ram_mb(), self._peak_ram_from_raw(raw_records)),
            reopen_time_s=0.0,
            timeout_policy_s=self.config.timeouts.measured_query_seconds,
            status=measurement.status,
            error_message=measurement.error_message,
            n_measured_queries=len(measurement.latencies_ms),
        )
        self._persist_run_metadata(artifacts, summary)
        return summary

    def run_workload_b2(
        self,
        engine_name: EngineName,
        records: Sequence[PreparedRecord],
        *,
        query_records: Sequence[PreparedRecord] | None = None,
        batch_size: int,
        ingest_rate_per_second: int,
        k: int,
        repetition_index: int = 0,
    ) -> BenchmarkSummary:
        """Execute workload B2 continuous ingest with periodic queries."""

        queries = list(query_records or self.default_query_records(records, query_count=8))
        artifacts = self._prepare_artifacts(engine_name, workload="b2")
        engine = self.engine_factory(engine_name, artifacts.engine_dir)
        raw_records: list[RawRecord] = []
        engine.reset()
        engine.connect()
        started_at = time.perf_counter()
        query_latencies_ms: list[float] = []
        query_recalls: list[float] = []
        cumulative_records: list[PreparedRecord] = []

        for batch_index, batch in enumerate(chunk_records(records, batch_size), start=0):
            engine_batch = [record_to_engine_record(record) for record in batch]
            batch_started_at = time.perf_counter()
            engine.insert(engine_batch)
            batch_elapsed = time.perf_counter() - batch_started_at
            cumulative_records.extend(batch)
            ram_after_mb = process_ram_mb()
            raw_records.append(
                RawRecord(
                    run_id=artifacts.run_id,
                    engine=engine_name,
                    workload="b2",
                    phase="ingest",
                    status="success",
                    timestamp_utc=utc_timestamp(),
                    repetition_index=repetition_index,
                    batch_index=batch_index,
                    batch_size=len(batch),
                    cumulative_vectors=len(cumulative_records),
                    batch_latency_ms=batch_elapsed * 1000,
                    throughput_vps=(len(batch) / batch_elapsed) if batch_elapsed else 0.0,
                    ram_mb=ram_after_mb,
                    ram_after_mb=ram_after_mb,
                )
            )
            stage_ground_truth = self._ground_truth_for_queries(cumulative_records, queries, k=k)
            for query_index, query_record in enumerate(queries):
                phase_started_at = time.perf_counter()
                hits = engine.search(
                    query_record.embedding,
                    k=k,
                    metric=self.config.engines.distance_metric,
                )
                latency_ms = (time.perf_counter() - phase_started_at) * 1000
                query_latencies_ms.append(latency_ms)
                recall_value = calculate_recall_at_k(
                    [hit.id for hit in hits],
                    stage_ground_truth.get(query_record.id, []),
                    k=k,
                )
                query_recalls.append(recall_value)
                returned_ids = [hit.id for hit in hits]
                raw_records.append(
                    RawRecord(
                        run_id=artifacts.run_id,
                        engine=engine_name,
                        workload="b2",
                        phase="query",
                        status="success",
                        timestamp_utc=utc_timestamp(),
                        repetition_index=repetition_index,
                        query_id=f"query-{query_index}",
                        query_index=query_index,
                        mode="serial",
                        latency_ms=latency_ms,
                        ram_mb=process_ram_mb(),
                        recall_at_k=recall_value,
                        returned_ids=returned_ids,
                        ground_truth_ids=stage_ground_truth.get(query_record.id, [])[:k],
                        is_warmup=False,
                    )
                )

        build_started_at = time.perf_counter()
        engine.build_index()
        build_time_s = time.perf_counter() - build_started_at
        engine.persist()
        write_raw_records(artifacts.raw_dir, raw_records)
        summary = self._build_summary(
            artifacts=artifacts,
            engine=engine,
            records=records,
            workload="b2",
            latencies_ms=query_latencies_ms,
            load_time_s=time.perf_counter() - started_at,
            build_time_s=build_time_s,
            k=k,
            batch_size=batch_size,
            recall_at_k=(sum(query_recalls) / len(query_recalls)) if query_recalls else 0.0,
            peak_ram_mb=max(process_ram_mb(), self._peak_ram_from_raw(raw_records)),
            reopen_time_s=0.0,
            timeout_policy_s=self.config.timeouts.ingest_batch_seconds,
            status=TimeoutStatus.SUCCESS.value,
            error_message=None,
            n_measured_queries=len(query_latencies_ms),
        )
        self._persist_run_metadata(artifacts, summary)
        return summary

    def run_workload_b3(
        self,
        engine_name: EngineName,
        records: Sequence[PreparedRecord],
        *,
        query_records: Sequence[PreparedRecord] | None = None,
        k: int,
        repetition_index: int = 0,
    ) -> BenchmarkSummary:
        """Execute workload B3 persist and reopen validation."""

        del repetition_index
        queries = list(query_records or self.default_query_records(records, query_count=8))
        artifacts = self._prepare_artifacts(engine_name, workload="b3")
        engine = self.engine_factory(engine_name, artifacts.engine_dir)
        raw_records: list[RawRecord] = []
        load_time_s, build_time_s = self._load_engine(engine, records)
        before_restart = {
            query.id: [
                hit.id
                for hit in engine.search(
                    query.embedding,
                    k,
                    self.config.engines.distance_metric,
                )
            ]
            for query in queries
        }
        reopen_started_at = time.perf_counter()
        reopen_result = execute_with_timeout(engine.reopen, self.config.timeouts.reopen_seconds)
        reopen_time_s = time.perf_counter() - reopen_started_at
        latencies_ms: list[float] = []
        recalls: list[float] = []
        for query_index, query in enumerate(queries):
            started_at = time.perf_counter()
            hits = engine.search(query.embedding, k, self.config.engines.distance_metric)
            latency_ms = (time.perf_counter() - started_at) * 1000
            latencies_ms.append(latency_ms)
            recall_value = calculate_recall_at_k(
                [hit.id for hit in hits],
                before_restart.get(query.id, []),
                k=k,
            )
            recalls.append(recall_value)
            raw_records.append(
                RawRecord(
                    run_id=artifacts.run_id,
                    engine=engine_name,
                    workload="b3",
                    phase="reopen_query",
                    status=reopen_result.status.value,
                    timestamp_utc=utc_timestamp(),
                    query_id=f"query-{query_index}",
                    latency_ms=latency_ms,
                    ram_mb=process_ram_mb(),
                    recall_at_k=recall_value,
                )
            )
        write_raw_records(artifacts.raw_dir, raw_records)
        summary = self._build_summary(
            artifacts=artifacts,
            engine=engine,
            records=records,
            workload="b3",
            latencies_ms=latencies_ms,
            load_time_s=load_time_s,
            build_time_s=build_time_s,
            k=k,
            batch_size=None,
            recall_at_k=(sum(recalls) / len(recalls)) if recalls else 0.0,
            peak_ram_mb=max(process_ram_mb(), self._peak_ram_from_raw(raw_records)),
            reopen_time_s=reopen_time_s,
            timeout_policy_s=self.config.timeouts.reopen_seconds,
            status=reopen_result.status.value,
            error_message=reopen_result.error_message,
            n_measured_queries=len(latencies_ms),
        )
        self._persist_run_metadata(artifacts, summary)
        return summary

    def run_workload_b4(
        self,
        engine_name: EngineName,
        records: Sequence[PreparedRecord],
        *,
        query_records: Sequence[PreparedRecord] | None = None,
        k: int,
        memory_limit_mb: float | None = None,
        repetition_index: int = 0,
    ) -> BenchmarkSummary:
        """Execute workload B4 under an optional memory threshold policy."""

        queries = list(query_records or self.default_query_records(records, query_count=16))
        artifacts = self._prepare_artifacts(engine_name, workload="b4")
        engine = self.engine_factory(engine_name, artifacts.engine_dir)
        raw_records: list[RawRecord] = []
        load_time_s, build_time_s = self._load_engine(engine, records)
        reopen_result = execute_with_timeout(engine.reopen, self.config.timeouts.reopen_seconds)
        latencies_ms: list[float] = []
        recalls: list[float] = []
        ground_truth = self._ground_truth_for_queries(records, queries, k=k)
        for query_index, query_record in enumerate(queries):
            started_at = time.perf_counter()
            hits = engine.search(query_record.embedding, k, self.config.engines.distance_metric)
            latency_ms = (time.perf_counter() - started_at) * 1000
            latencies_ms.append(latency_ms)
            recall_value = calculate_recall_at_k(
                [hit.id for hit in hits],
                ground_truth.get(query_record.id, []),
                k=k,
            )
            recalls.append(recall_value)
            raw_records.append(
                RawRecord(
                    run_id=artifacts.run_id,
                    engine=engine_name,
                    workload="b4",
                    phase="query",
                    status=reopen_result.status.value,
                    timestamp_utc=utc_timestamp(),
                    repetition_index=repetition_index,
                    query_id=f"query-{query_index}",
                    query_index=query_index,
                    mode="serial",
                    latency_ms=latency_ms,
                    ram_mb=process_ram_mb(),
                    recall_at_k=recall_value,
                    returned_ids=[hit.id for hit in hits],
                    ground_truth_ids=ground_truth.get(query_record.id, [])[:k],
                    is_warmup=False,
                    error=None,
                )
            )
        peak_ram_mb = max(process_ram_mb(), self._peak_ram_from_raw(raw_records))
        status = reopen_result.status.value
        if memory_limit_mb is not None and peak_ram_mb > memory_limit_mb:
            status = TimeoutStatus.PARTIAL.value
        write_raw_records(artifacts.raw_dir, raw_records)
        summary = self._build_summary(
            artifacts=artifacts,
            engine=engine,
            records=records,
            workload="b4",
            latencies_ms=latencies_ms,
            load_time_s=load_time_s,
            build_time_s=build_time_s,
            k=k,
            batch_size=None,
            recall_at_k=(sum(recalls) / len(recalls)) if recalls else 0.0,
            peak_ram_mb=peak_ram_mb,
            reopen_time_s=0.0,
            timeout_policy_s=self.config.timeouts.reopen_seconds,
            status=status,
            error_message=reopen_result.error_message,
            n_measured_queries=len(latencies_ms),
        )
        self._persist_run_metadata(artifacts, summary)
        return summary

    def _prepare_artifacts(self, engine_name: EngineName, *, workload: str) -> RunArtifacts:
        run_id = new_run_id()
        # Use project results dir if available, otherwise use config's benchmark_dir
        if self.project:
            results_base = self.project.results_dir
        else:
            results_base = self.config.paths.benchmark_dir / "results"
        run_dir = results_base / engine_name / workload / run_id
        raw_dir = run_dir
        engine_dir = run_dir / "engine"
        log_path = run_dir / "benchmark.log"
        raw_dir.mkdir(parents=True, exist_ok=True)
        engine_dir.mkdir(parents=True, exist_ok=True)
        return RunArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            raw_dir=raw_dir,
            engine_dir=engine_dir,
            log_path=log_path,
        )

    def _get_run_logger(self, artifacts: RunArtifacts) -> logging.Logger:
        """Return an isolated file logger writing to the run directory."""
        logger = logging.getLogger(artifacts.run_id)
        if not logger.handlers:
            handler = logging.FileHandler(artifacts.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
        return logger

    def _build_engine(self, engine_name: EngineName, storage_path: Path) -> BenchmarkEngine:
        if engine_name == "faiss":
            return FaissEngine(
                storage_path,
                distance_metric=self.config.engines.distance_metric,
                index_type=self.config.engines.faiss_index_type,
                nlist=self.config.engines.faiss_nlist,
            )
        if engine_name == "chromadb":
            return ChromaEngine(storage_path, distance_metric=self.config.engines.distance_metric)
        return LanceDbEngine(storage_path, distance_metric=self.config.engines.distance_metric)

    def _load_engine(
        self,
        engine: BenchmarkEngine,
        records: Sequence[PreparedRecord],
    ) -> tuple[float, float]:
        engine.reset()
        engine.connect()
        load_started_at = time.perf_counter()
        engine.insert(record_to_engine_record(record) for record in records)
        load_time_s = time.perf_counter() - load_started_at
        build_started_at = time.perf_counter()
        engine.build_index()
        build_time_s = time.perf_counter() - build_started_at
        engine.persist()
        return load_time_s, build_time_s

    def _ground_truth_for_queries(
        self,
        base_records: Sequence[PreparedRecord],
        query_records: Sequence[PreparedRecord],
        *,
        k: int,
    ) -> dict[str, list[str]]:
        if not base_records or not query_records:
            return {}
        rows = compute_exact_ground_truth(
            [record.id for record in base_records],
            [record.embedding for record in base_records],
            [record.id for record in query_records],
            [record.embedding for record in query_records],
            k=k,
        )
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row.query_id, []).append(row.neighbor_id)
        return result

    def _persist_run_metadata(self, artifacts: RunArtifacts, summary: BenchmarkSummary) -> None:
        write_summary(artifacts.run_dir, summary)
        write_env_snapshot(
            artifacts.run_dir,
            {
                "config": self.config.model_dump(mode="json"),
                "host_fingerprint": self.host_fingerprint.to_dict(),
                "run_id": artifacts.run_id,
            },
        )
        if self.project:
            logger.info(
                "Project-aware write: run_id=%s to %s",
                artifacts.run_id,
                artifacts.run_dir,
            )
        log = self._get_run_logger(artifacts)
        log.info(
            "completed run_id=%s engine=%s workload=%s status=%s "
            "latency_ms=%.3f recall=%.4f qps=%.2f peak_ram_mb=%.1f",
            summary.run_id,
            summary.engine,
            summary.workload,
            summary.status,
            summary.mean_latency_ms,
            summary.recall_at_k or 0.0,
            summary.qps,
            summary.peak_ram_mb,
        )
        for handler in log.handlers[:]:
            handler.close()
            log.removeHandler(handler)

    def _build_summary(
        self,
        *,
        artifacts: RunArtifacts,
        engine: BenchmarkEngine,
        records: Sequence[PreparedRecord],
        workload: str,
        latencies_ms: Sequence[float],
        load_time_s: float,
        build_time_s: float,
        k: int | None,
        batch_size: int | None,
        recall_at_k: float | None,
        peak_ram_mb: float,
        reopen_time_s: float,
        timeout_policy_s: int,
        status: str,
        error_message: str | None,
        n_measured_queries: int,
        filter_selectivity: float | None = None,
    ) -> BenchmarkSummary:
        latency_stats = summarize_latencies(latencies_ms)
        runtime = engine.describe_runtime()
        dimension = len(records[0].embedding) if records else 0
        return BenchmarkSummary(
            run_id=artifacts.run_id,
            timestamp=utc_timestamp(),
            engine=runtime.engine,
            engine_version=runtime.package_version,
            dataset_name=self.config.dataset.primary,
            dataset_group="rag" if self.config.dataset.primary.endswith("wikipedia") else "ann",
            dataset_size=len(records),
            dimension=dimension,
            index_type=runtime.index_type,
            distance_metric=runtime.distance_metric,
            workload=workload,
            filter_selectivity=filter_selectivity,
            k=k,
            batch_size=batch_size,
            index_params={"faiss_nlist": self.config.engines.faiss_nlist},
            search_params={},
            n_warmup_queries=min(self.config.workloads.warmup_queries, len(latencies_ms)),
            n_measured_queries=n_measured_queries,
            n_repetitions=self.config.workloads.repetitions,
            load_time_s=load_time_s,
            build_time_s=build_time_s,
            time_to_serve_s=load_time_s + build_time_s,
            mean_latency_ms=latency_stats.mean_ms,
            p95_latency_ms=latency_stats.p95_ms,
            p99_latency_ms=latency_stats.p99_ms,
            std_latency_ms=latency_stats.std_ms,
            qps=queries_per_second(n_measured_queries, sum(latencies_ms) / 1000),
            recall_at_k=recall_at_k,
            peak_ram_mb=peak_ram_mb,
            disk_size_mb=float(engine.disk_size() / (1024**2)),
            reopen_time_s=reopen_time_s,
            timeout_policy_s=timeout_policy_s,
            status=status,
            error_message=error_message,
            host_fingerprint=self.host_fingerprint.to_dict(),
            raw_data_path=f"raw_queries_{artifacts.run_id}.parquet",
        )

    def _run_query_phase(
        self,
        *,
        engine: BenchmarkEngine,
        engine_name: EngineName,
        workload: str,
        run_id: str,
        queries: Sequence[PreparedRecord],
        selectivity_records: Sequence[PreparedRecord] | None = None,
        raw_records: list[RawRecord],
        ground_truth: Mapping[str, list[str]],
        k: int,
        filters: MetadataFilter | None,
        repetition_index: int,
    ) -> "QueryMeasurement":
        latencies_ms: list[float] = []
        recalls: list[float] = []
        peak_ram_mb = process_ram_mb()

        def warmup_phase() -> None:
            for query in queries[: self.config.workloads.warmup_queries]:
                if filters is None:
                    engine.search(query.embedding, k, self.config.engines.distance_metric)
                else:
                    engine.search_filtered(
                        query.embedding,
                        filters,
                        k,
                        self.config.engines.distance_metric,
                    )

        warmup_result = execute_with_timeout(
            warmup_phase,
            self.config.timeouts.warmup_query_seconds,
            timeout_status=TimeoutStatus.PARTIAL,
        )
        if warmup_result.status == TimeoutStatus.ERROR:
            return QueryMeasurement(
                latencies_ms=[],
                mean_recall=0.0,
                peak_ram_mb=peak_ram_mb,
                status=warmup_result.status.value,
                error_message=warmup_result.error_message,
            )

        def measured_phase() -> None:
            nonlocal peak_ram_mb
            for query_index, query in enumerate(queries):
                started_at = time.perf_counter()
                if filters is None:
                    hits = engine.search(query.embedding, k, self.config.engines.distance_metric)
                else:
                    hits = engine.search_filtered(
                        query.embedding,
                        filters,
                        k,
                        self.config.engines.distance_metric,
                    )
                latency_ms = (time.perf_counter() - started_at) * 1000
                returned_ids = [hit.id for hit in hits]
                recall_value = calculate_recall_at_k(
                    returned_ids,
                    ground_truth.get(query.id, []),
                    k=k,
                )
                latencies_ms.append(latency_ms)
                recalls.append(recall_value)
                peak_ram_mb = max(peak_ram_mb, process_ram_mb())
                raw_records.append(
                    RawRecord(
                        run_id=run_id,
                        engine=engine_name,
                        workload=workload,
                        phase="query",
                        status="success",
                        timestamp_utc=utc_timestamp(),
                        repetition_index=repetition_index,
                        query_id=f"query-{query_index}",
                        query_index=query_index,
                        mode="serial",
                        latency_ms=latency_ms,
                        ram_mb=peak_ram_mb,
                        recall_at_k=recall_value,
                        returned_ids=returned_ids,
                        ground_truth_ids=ground_truth.get(query.id, [])[:k],
                        filter_applied=(
                            None if filters is None else self._filter_name(filters, engine)
                        ),
                        filter_selectivity_observed=(
                            self._observed_selectivity(
                                filters,
                                selectivity_records if selectivity_records is not None else queries,
                            )
                            if filters is not None
                            else None
                        ),
                        is_warmup=False,
                    )
                )

        measured_result = execute_with_timeout(
            measured_phase,
            self.config.timeouts.measured_query_seconds,
            timeout_status=TimeoutStatus.PARTIAL,
        )
        status = measured_result.status.value
        if warmup_result.status == TimeoutStatus.PARTIAL and status == TimeoutStatus.SUCCESS.value:
            status = TimeoutStatus.PARTIAL.value
        return QueryMeasurement(
            latencies_ms=latencies_ms,
            mean_recall=(sum(recalls) / len(recalls)) if recalls else 0.0,
            peak_ram_mb=peak_ram_mb,
            status=status,
            error_message=measured_result.error_message or warmup_result.error_message,
        )

    @staticmethod
    def _filter_name(filters: MetadataFilter, engine: BenchmarkEngine) -> str:
        if engine.name == "faiss":
            return "pre-filter:application-layer"
        return f"native:{len(filters.clauses)}-clauses"

    @staticmethod
    def _observed_selectivity(
        filters: MetadataFilter,
        records: Sequence[PreparedRecord],
    ) -> float:
        if not records:
            return 0.0
        matched = sum(1 for record in records if filters.matches(record_to_metadata(record)))
        return float(matched / len(records))

    @staticmethod
    def _peak_ram_from_raw(raw_records: Sequence[RawRecord]) -> float:
        ram_values = [record.ram_mb for record in raw_records if record.ram_mb is not None]
        return max(ram_values) if ram_values else 0.0


@dataclass(slots=True, frozen=True)
class QueryMeasurement:
    """Internal accumulator for query workload execution."""

    latencies_ms: list[float]
    mean_recall: float
    peak_ram_mb: float
    status: str
    error_message: str | None


def chunk_records(
    records: Sequence[PreparedRecord],
    batch_size: int,
) -> list[list[PreparedRecord]]:
    """Split prepared records into fixed-size batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        list(records[index : index + batch_size])
        for index in range(0, len(records), batch_size)
    ]


def record_to_engine_record(record: PreparedRecord) -> EngineRecord:
    """Convert a prepared dataset record into the canonical engine payload."""

    return EngineRecord(
        id=record.id,
        embedding=list(record.embedding),
        document=record.text,
        metadata=record_to_metadata(record),
    )


def record_to_metadata(record: PreparedRecord) -> dict[str, str | int | float | bool]:
    """Extract metadata fields used by filtered workloads."""

    metadata: dict[str, str | int | float | bool] = {
        "source_dataset": record.source_dataset,
        "document_title": record.document_title,
        "document_length_bucket": record.document_length_bucket,
        "categoria": record.categoria,
        "tag": record.tag,
        "autor": record.autor,
        "data_criacao": record.data_criacao,
    }
    if record.parent_document_id is not None:
        metadata["parent_document_id"] = record.parent_document_id
    if record.token_count is not None:
        metadata["token_count"] = record.token_count
    return metadata
