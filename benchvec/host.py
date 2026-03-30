"""Host fingerprint collection utilities required by PRD03."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass(slots=True)
class HostFingerprint:
    """Serializable host metadata for reproducible benchmark runs."""

    os_name: str
    os_version: str
    kernel_version: str
    python_version: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_frequency_mhz: float | None
    ram_total_gb: float
    storage_type: str
    storage_detection_method: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the fingerprint into a dictionary."""

        return asdict(self)


def _read_linux_rotational(device_name: str) -> str | None:
    rotational_path = Path("/sys/block") / device_name / "queue" / "rotational"
    if not rotational_path.exists():
        return None
    rotational_value = rotational_path.read_text(encoding="utf-8").strip()
    return "hdd" if rotational_value == "1" else "ssd"


def _detect_storage_type() -> tuple[str, str]:
    """Best-effort local storage detection across platforms."""

    if platform.system() == "Linux":
        partition = psutil.disk_partitions(all=False)[0]
        device_name = Path(partition.device).name.rstrip("0123456789")
        rotational_type = _read_linux_rotational(device_name)
        if rotational_type is not None:
            return rotational_type, "linux_sysfs"
        return "unknown", "linux_partition_only"

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["diskutil", "info", "/"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "unknown", "darwin_unavailable"

        lowered_output = result.stdout.lower()
        if "solid state" in lowered_output or "ssd" in lowered_output or "apfs" in lowered_output:
            return "ssd", "diskutil"
        return "unknown", "diskutil"

    return "unknown", "unsupported_platform"


def collect_host_fingerprint() -> HostFingerprint:
    """Collect host environment metadata with a stable schema."""

    frequency = psutil.cpu_freq()
    storage_type, detection_method = _detect_storage_type()
    return HostFingerprint(
        os_name=platform.system(),
        os_version=platform.platform(),
        kernel_version=platform.release(),
        python_version=platform.python_version(),
        cpu_model=platform.processor() or platform.machine() or "unknown",
        cpu_physical_cores=psutil.cpu_count(logical=False) or 0,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 0,
        cpu_frequency_mhz=frequency.max if frequency is not None else None,
        ram_total_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        storage_type=storage_type,
        storage_detection_method=detection_method,
    )
