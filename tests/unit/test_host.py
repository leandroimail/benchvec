"""Unit tests for host fingerprint collection."""

from __future__ import annotations

import pytest

from benchvec.host import collect_host_fingerprint


@pytest.mark.unit
def test_collect_host_fingerprint_has_required_fields() -> None:
    fingerprint = collect_host_fingerprint()
    payload = fingerprint.to_dict()

    assert payload["os_name"]
    assert payload["kernel_version"]
    assert fingerprint.ram_total_gb > 0
    assert fingerprint.cpu_logical_cores >= fingerprint.cpu_physical_cores
    assert fingerprint.storage_type in {"ssd", "hdd", "unknown"}
