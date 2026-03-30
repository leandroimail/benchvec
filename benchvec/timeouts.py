"""Timeout helpers for benchmark phase execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Generic, TypeVar

ResultType = TypeVar("ResultType")


class TimeoutStatus(StrEnum):
    """Standardized execution statuses required by the PRD."""

    SUCCESS = "success"
    TIMED_OUT = "timed_out"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(slots=True)
class TimeoutResult(Generic[ResultType]):
    """Result container for operations executed with a timeout."""

    status: TimeoutStatus
    elapsed_seconds: float
    payload: ResultType | None = None
    error_message: str | None = None


def execute_with_timeout(
    operation: Callable[[], ResultType],
    timeout_seconds: int,
    *,
    timeout_status: TimeoutStatus = TimeoutStatus.TIMED_OUT,
) -> TimeoutResult[ResultType]:
    """Execute an operation in a worker thread with bounded waiting time."""

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        try:
            payload = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            return TimeoutResult(
                status=timeout_status,
                elapsed_seconds=time.perf_counter() - started_at,
                error_message=f"Operation exceeded timeout of {timeout_seconds} seconds",
            )
        except Exception as exc:  # pragma: no cover - exercised through tests with explicit failure
            return TimeoutResult(
                status=TimeoutStatus.ERROR,
                elapsed_seconds=time.perf_counter() - started_at,
                error_message=str(exc),
            )

    return TimeoutResult(
        status=TimeoutStatus.SUCCESS,
        elapsed_seconds=time.perf_counter() - started_at,
        payload=payload,
    )
