from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytest

from benchvec.config import BenchmarkConfig
from benchvec.datasets import PreparedRecord
from benchvec.engines.base import (
    BenchmarkEngine,
    DistanceMetric,
    EngineRecord,
    EngineRuntime,
    MetadataFilter,
    SearchHit,
)
from benchvec.runner import BenchmarkRunner


class E2EFakeEngine(BenchmarkEngine):
    def __init__(self, storage_path: Path) -> None:
        self.name = "fake"
        self.distance_metric = "l2"
        self.index_type = "flat"
        self.native_filtering = True
        self.storage_path = storage_path
        self._records: list[EngineRecord] = []
        self._persisted_records: list[EngineRecord] = []

    def connect(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self._records = []
        self._persisted_records = []
        if self.storage_path.exists():
            for item in self.storage_path.iterdir():
                if item.is_file():
                    item.unlink()

    def insert(self, batch: Iterable[EngineRecord]) -> None:
        self._records.extend(batch)

    def build_index(self) -> None:
        return None

    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        return self._search_records(self._records, query, k)

    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metric, params
        filtered = [record for record in self._records if filters.matches(record.metadata)]
        return self._search_records(filtered, query, k)

    def delete(self, ids: Sequence[str]) -> int:
        id_set = set(ids)
        before = len(self._records)
        self._records = [record for record in self._records if record.id not in id_set]
        return before - len(self._records)

    def upsert(self, records: Sequence[EngineRecord]) -> None:
        materialized = list(records)
        self.delete([record.id for record in materialized])
        self._records.extend(materialized)

    def persist(self) -> None:
        self._persisted_records = list(self._records)
        (self.storage_path / "state.txt").write_text(str(len(self._records)), encoding="utf-8")

    def reopen(self) -> None:
        self._records = list(self._persisted_records)

    def disk_size(self) -> int:
        state_path = self.storage_path / "state.txt"
        return state_path.stat().st_size if state_path.exists() else 0

    def describe_runtime(self) -> EngineRuntime:
        return EngineRuntime(
            engine=self.name,
            package_version="test",
            distance_metric="l2",
            index_type="flat",
            native_filtering=True,
            storage_path=str(self.storage_path),
        )

    @staticmethod
    def _search_records(records: list[EngineRecord], query: list[float], k: int) -> list[SearchHit]:
        ranked = sorted(
            records,
            key=lambda record: sum(
                (left - right) ** 2 for left, right in zip(record.embedding, query)
            ),
        )
        return [
            SearchHit(
                id=record.id,
                score=float(
                    sum((left - right) ** 2 for left, right in zip(record.embedding, query))
                ),
                document=record.document,
                metadata=dict(record.metadata),
            )
            for record in ranked[:k]
        ]


def _prepared_records(count: int = 20) -> list[PreparedRecord]:
    return [
        PreparedRecord(
            id=f"doc-{index}",
            text=f"Document {index}",
            embedding=[float(index), float(index % 2), 0.0],
            source_dataset="fast_wikipedia",
            document_title=f"Doc {index}",
            document_length_bucket="short",
            categoria="data_engineering" if index < (count // 2) else "science",
            tag="engenharia_dados" if index < (count // 2) else "ciencia",
            autor="engenharia" if index % 2 == 0 else "pesquisa",
            data_criacao=1704067200 + (index % 5) * 86400,
            token_count=32 + index,
        )
        for index in range(count)
    ]


@pytest.mark.unit
def test_e2e_fast_profile_runs_all_workloads(tmp_path: Path) -> None:
    config = BenchmarkConfig()
    config.paths.benchmark_dir = tmp_path / "benchmarks"
    config.workloads.warmup_queries = 1
    config.workloads.repetitions = 1
    config.dataset.ground_truth_query_count = 4

    runner = BenchmarkRunner(
        config,
        engine_factory=lambda name, path: E2EFakeEngine(path),
    )
    records = _prepared_records()
    queries = runner.default_query_records(records)
    filter_scenarios = runner.build_filter_scenarios(records)
    filters = filter_scenarios[0].filters
    batch_size = min(1000, len(records))

    workloads = [
        runner.run_workload_a1("faiss", records, batch_size=batch_size),
        runner.run_workload_a2("faiss", records, query_records=queries, k=2),
        runner.run_workload_b1("faiss", records, filters=filters, query_records=queries, k=2),
        runner.run_workload_b2(
            "faiss",
            records,
            query_records=queries[:2],
            batch_size=batch_size,
            ingest_rate_per_second=10,
            k=2,
        ),
        runner.run_workload_b3("faiss", records, query_records=queries[:2], k=2),
        runner.run_workload_b4("faiss", records, query_records=queries, k=2),
    ]

    assert [summary.workload for summary in workloads] == ["a1", "a2", "b1", "b2", "b3", "b4"]
    assert all(summary.run_id for summary in workloads)
    recall_values = [
        summary.recall_at_k for summary in workloads if summary.recall_at_k is not None
    ]
    assert recall_values
    assert all(0.0 <= recall <= 1.0 for recall in recall_values)
