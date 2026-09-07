"""Common engine contracts and filter helpers for BenchVec."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

MetadataValue = str | int | float | bool
FilterOperator = Literal["eq", "gt", "gte", "lt", "lte"]
DistanceMetric = Literal["l2", "cosine"]


def package_version(package_name: str) -> str:
    """Resolve the installed package version for runtime reporting."""

    return version(package_name)


def directory_size_bytes(path: Path) -> int:
    """Calculate the total size for a directory tree or a single file."""

    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass(slots=True, frozen=True)
class FilterClause:
    """Single metadata predicate used in filtered search workloads."""

    field: str
    operator: FilterOperator
    value: MetadataValue


@dataclass(slots=True, frozen=True)
class MetadataFilter:
    """Metadata filter expression shared across all engine adapters."""

    clauses: tuple[FilterClause, ...] = ()

    def matches(self, metadata: Mapping[str, MetadataValue]) -> bool:
        """Evaluate the filter against an in-memory metadata mapping."""

        for clause in self.clauses:
            if clause.field not in metadata:
                return False
            current_value = metadata[clause.field]
            if clause.operator == "eq" and current_value != clause.value:
                return False
            if clause.operator != "eq":
                if isinstance(current_value, bool) or isinstance(clause.value, bool):
                    return False
                if isinstance(current_value, str) and isinstance(clause.value, str):
                    if clause.operator == "gt" and current_value <= clause.value:
                        return False
                    if clause.operator == "gte" and current_value < clause.value:
                        return False
                    if clause.operator == "lt" and current_value >= clause.value:
                        return False
                    if clause.operator == "lte" and current_value > clause.value:
                        return False
                    continue
                if isinstance(current_value, (int, float)) and isinstance(
                    clause.value,
                    (int, float),
                ):
                    left_value = float(current_value)
                    right_value = float(clause.value)
                    if clause.operator == "gt" and left_value <= right_value:
                        return False
                    if clause.operator == "gte" and left_value < right_value:
                        return False
                    if clause.operator == "lt" and left_value >= right_value:
                        return False
                    if clause.operator == "lte" and left_value > right_value:
                        return False
                    continue
                return False
        return True

    def to_chroma_where(self) -> dict[str, Any] | None:
        """Convert the filter to the ChromaDB `where` format."""

        if not self.clauses:
            return None

        conditions: list[dict[str, Any]] = []
        operator_map = {
            "eq": "$eq",
            "gt": "$gt",
            "gte": "$gte",
            "lt": "$lt",
            "lte": "$lte",
        }
        for clause in self.clauses:
            if clause.operator == "eq":
                conditions.append({clause.field: clause.value})
            else:
                conditions.append({clause.field: {operator_map[clause.operator]: clause.value}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def to_lancedb_where(self) -> str | None:
        """Convert the filter to a LanceDB SQL-like where expression."""

        if not self.clauses:
            return None

        operator_map = {
            "eq": "=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
        }

        def sql_literal(value: MetadataValue) -> str:
            if isinstance(value, bool):
                return "TRUE" if value else "FALSE"
            if isinstance(value, (int, float)):
                return repr(value)
            # Single-quote SQL literals with '' doubling — json.dumps-style
            # double quotes break the lance SQL parser on special characters.
            return "'" + str(value).replace("'", "''") + "'"

        expressions = []
        for clause in self.clauses:
            expressions.append(
                f"{clause.field} {operator_map[clause.operator]} {sql_literal(clause.value)}"
            )
        return " AND ".join(expressions)


@dataclass(slots=True)
class EngineRecord:
    """Canonical vector record passed into engine adapters."""

    id: str
    embedding: list[float]
    document: str | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    """Single vector search hit returned by an engine adapter."""

    id: str
    score: float
    document: str | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(slots=True)
class EngineRuntime:
    """Serializable runtime description for one engine instance."""

    engine: str
    package_version: str
    distance_metric: DistanceMetric
    index_type: str
    native_filtering: bool
    storage_path: str


class BenchmarkEngine(ABC):
    """Abstract interface required by PRD03 for all benchmark engines."""

    name: str
    distance_metric: DistanceMetric
    index_type: str
    native_filtering: bool
    # Search parameters actually used by the most recent search call
    # (populated by adapters; captured into BenchmarkSummary.search_params).
    last_search_params: dict[str, Any]

    @abstractmethod
    def connect(self) -> None:
        """Initialize engine runtime state."""

    @abstractmethod
    def reset(self) -> None:
        """Clear engine state and remove persisted data for a clean run."""

    @abstractmethod
    def insert(self, batch: Iterable[EngineRecord]) -> None:
        """Insert a batch of records into the engine."""

    @abstractmethod
    def build_index(self) -> None:
        """Finalize any deferred indexing work after inserts."""

    @abstractmethod
    def search(
        self,
        query: list[float],
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Execute an unfiltered vector search."""

    @abstractmethod
    def search_filtered(
        self,
        query: list[float],
        filters: MetadataFilter,
        k: int,
        metric: DistanceMetric,
        params: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Execute a metadata-filtered vector search."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Remove records by id.

        Returns the number removed when the backend reports it; best-effort
        count otherwise (some embedded APIs do not report matches).
        """

    @abstractmethod
    def upsert(self, records: Sequence[EngineRecord]) -> None:
        """Insert records, overwriting any with the same id."""

    @abstractmethod
    def persist(self) -> None:
        """Flush in-memory state to durable local storage."""

    @abstractmethod
    def reopen(self) -> None:
        """Reopen an already persisted engine state."""

    @abstractmethod
    def disk_size(self) -> int:
        """Return the persisted footprint in bytes."""

    @abstractmethod
    def describe_runtime(self) -> EngineRuntime:
        """Describe the current runtime for reproducibility records."""
