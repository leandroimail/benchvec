"""Unit tests for timeout execution helpers."""

from __future__ import annotations

import time

import pytest

from benchvec.timeouts import TimeoutStatus, execute_with_timeout


@pytest.mark.unit
def test_execute_with_timeout_success() -> None:
    result = execute_with_timeout(lambda: "ok", timeout_seconds=1)

    assert result.status == TimeoutStatus.SUCCESS
    assert result.payload == "ok"
    assert result.error_message is None


@pytest.mark.unit
def test_execute_with_timeout_timeout() -> None:
    result = execute_with_timeout(lambda: time.sleep(0.2), timeout_seconds=0)

    assert result.status == TimeoutStatus.TIMED_OUT
    assert result.payload is None
    assert result.error_message is not None


@pytest.mark.unit
def test_execute_with_timeout_error() -> None:
    def failing_operation() -> str:
        raise RuntimeError("boom")

    result = execute_with_timeout(failing_operation, timeout_seconds=1)

    assert result.status == TimeoutStatus.ERROR
    assert result.error_message == "boom"
