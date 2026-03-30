"""Pytest configuration for BenchVec tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register project markers used by the test suite."""

    config.addinivalue_line("markers", "unit: fast isolated tests")
