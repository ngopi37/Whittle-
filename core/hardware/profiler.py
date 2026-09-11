"""Measured local hardware profiling."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas.hardware import CpuProfile, GpuProfile, HardwareProfile, MemoryProfile, StorageProfile

_POWERSHELL_TIMEOUT_S = 6

# Windows PF_* processor feature identifiers (winnt.h), mapped to normalized names.
_PROCESSOR_FEATURES: dict[int, str] = {
    6: "sse",       # PF_XMMI_INSTRUCTIONS_AVAILABLE
    10: "sse2",     # PF_XMMI64_INSTRUCTIONS_AVAILABLE
    13: "sse3",     # PF_SSE3_INSTRUCTIONS_AVAILABLE
    36: "ssse3",    # PF_SSSE3_INSTRUCTIONS_AVAILABLE
    37: "sse4_1",   # PF_SSE4_1_INSTRUCTIONS_AVAILABLE
    38: "sse4_2",   # PF_SSE4_2_INSTRUCTIONS_AVAILABLE
    39: "avx",      # PF_AVX_INSTRUCTIONS_AVAILABLE
    40: "avx2",     # PF_AVX2_INSTRUCTIONS_AVAILABLE
    41: "avx512f",  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
}

# Display adapter class GUID under HKLM\SYSTEM\CurrentControlSet\Control\Class.
_DISPLAY_CLASS_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
)


@dataclass(frozen=True)
class WindowsCpuInfo:
    """Small local snapshot from Windows processor metadata."""

    vendor: str
    model: str
    architecture: str | None
    max_clock_mhz: float | None


class HardwareProfiler:
    """Detect hardware using operating-system APIs and optional local probes."""

    def profile(self) -> HardwareProfile:
        """Return a normalized snapshot of the current machine."""
        import psutil

        windows_data = _windows_hardware_data() if platform.system() == "Windows" else {}
        cpu_info = _parse_windows_cpu(windows_data.get("cpu"))
        frequency = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = shutil.disk_usage(_profiled_disk_root())
        return HardwareProfile(
            os=platform.system() or "Unknown",
            architecture=_normalize_architecture(
                cpu_info.architecture if cpu_info and cpu_info.architecture else platform.machine()
            ),
            cpu=CpuProfile(
                vendor=_cpu_vendor(cpu_info),
                model=_cpu_model(cpu_info),
                physical_cores=psutil.cpu_count(logical=False) or 1,
                logical_cores=psutil.cpu_count(logical=True) or 1,
                frequency_mhz=_cpu_frequency_mhz(cpu_info, frequency.max if frequency else None),
                instruction_sets=_detect_instruction_sets(),
            ),
            memory=MemoryProfile(
                total_gb=memory.total / 2**30, available_gb=memory.available / 2**30
            ),
            gpu=_select_gpu(windows_data.get("gpus")),
            storage=StorageProfile(capacity_gb=disk.total / 2**30, free_gb=disk.free / 2**30),
        )


def _profiled_disk_root() -> str:
    """Return the drive/root that contains the current workspace."""
    anchor = Path.cwd().anchor
    return anchor if anchor else "/"


def _normalize_architecture(value: str | None) -> str:
    """Normalize common architecture names into stable schema values."""
    normalized = (value or "").strip().lower()
    if normalized in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    return value or "Unknown"


def _powershell_executable() -> str | None:
    """Resolve a PowerShell executable, preferring the in-box Windows PowerShell."""
    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    candidates = [
        str(Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"),
        "powershell",
        "pwsh",
    ]
    for candidate in candidates:
        if Path(candidate).is_file() or shutil.which(candidate):
            return candidate
    return None


def _run_powershell(script: str) -> Any:
    """Run a constant PowerShell script and return parsed JSON, or ``None`` on failure."""
    executable = _powershell_executable()
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=_POWERSHELL_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _windows_hardware_data() -> dict[str, Any]:
    """Fetch CPU and video-controller metadata in a single PowerShell invocation."""
    script = (
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 "
        "Manufacturer,Name,Architecture,MaxClockSpeed;"
        "$gpus = Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name } | "
        "Select-Object Name,AdapterRAM,AdapterCompatibility,PNPDeviceID;"
        "[pscustomobject]@{ cpu = $cpu; gpus = @($gpus) } | ConvertTo-Json -Depth 4 -Compress"
    )
    raw = _run_powershell(script)
    return raw if isinstance(raw, dict) else {}


def _parse_windows_cpu(raw: object) -> WindowsCpuInfo | None:
    """Normalize the CPU section of the combined Windows metadata payload."""
    if not isinstance(raw, dict):
        return None
    architecture = _windows_architecture_name(raw.get("Architecture"))
    max_clock = raw.get("MaxClockSpeed")
    return WindowsCpuInfo(
        vendor=str(raw.get("Manufacturer") or "Unknown"),
        model=str(raw.get("Name") or "Unknown"),
        architecture=architecture,
        max_clock_mhz=float(max_clock) if isinstance(max_clock, int | float) else None,
    )


def _windows_architecture_name(code: object) -> str | None:
    """Map Win32_Processor architecture codes into normalized names."""
    names = {0: "x86", 5: "arm", 9: "x86_64", 12: "arm64"}
    return names.get(code) if isinstance(code, int) else None


def _cpu_vendor(cpu_info: WindowsCpuInfo | None) -> str:
    """Return a CPU vendor using local OS metadata and environment fallback."""
    if cpu_info and cpu_info.vendor != "Unknown":
        return cpu_info.vendor
    identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
    if "genuineintel" in identifier.lower():
        return "GenuineIntel"
    if "authenticamd" in identifier.lower():
        return "AuthenticAMD"
    return platform.processor() or "Unknown"


def _cpu_model(cpu_info: WindowsCpuInfo | None) -> str:
    """Return the local CPU model string when the OS exposes it."""
    if cpu_info and cpu_info.model != "Unknown":
        return cpu_info.model
    return platform.processor() or platform.machine() or "Unknown"


def _cpu_frequency_mhz(
    cpu_info: WindowsCpuInfo | None, psutil_frequency: float | None
) -> float | None:
    """Prefer measured psutil frequency and fall back to Windows max clock metadata."""
    if psutil_frequency and psutil_frequency > 0:
        return float(psutil_frequency)
    if cpu_info and cpu_info.max_clock_mhz:
        return cpu_info.max_clock_mhz
    return None


def _detect_instruction_sets() -> list[str]:
    """Detect a conservative subset of local CPU capabilities."""
    instruction_sets: list[str] = []
    if platform.machine().lower() in {"amd64", "x86_64"}:
        instruction_sets.extend(["x86_64", "sse2"])
    if platform.system() == "Windows":
        instruction_sets.extend(_windows_processor_features())
    return sorted(set(instruction_sets))


def _windows_processor_features() -> list[str]:
    """Use IsProcessorFeaturePresent for features Windows exposes directly."""
    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        return []
    probe = kernel32.IsProcessorFeaturePresent
    probe.argtypes = [wintypes.DWORD]
    probe.restype = wintypes.BOOL
    detected: list[str] = []
    for feature_id, name in _PROCESSOR_FEATURES.items():
        try:
            if probe(feature_id):
                detected.append(name)
        except OSError:
            continue
    return detected


def _select_gpu(raw_gpus: object) -> GpuProfile:
    """Pick the most capable local display adapter from Windows metadata."""
    if not isinstance(raw_gpus, list):
        return GpuProfile()
    parsed = [_gpu_from_raw(entry) for entry in raw_gpus if isinstance(entry, dict)]
    candidates = [gpu for gpu in parsed if gpu is not None]
    if not candidates:
        return GpuProfile()
    candidates.sort(key=_gpu_rank, reverse=True)
    return candidates[0]


def _gpu_from_raw(entry: dict[str, Any]) -> GpuProfile | None:
    """Build a GpuProfile from one Win32_VideoController record."""
    name = str(entry.get("Name") or "").strip()
    if not name:
        return None
    lowered = name.lower()
    if "microsoft basic display" in lowered or "remote display" in lowered:
        return None
    adapter_ram = entry.get("AdapterRAM")
    adapter_ram_bytes = float(adapter_ram) if isinstance(adapter_ram, int | float) else 0.0
    registry_bytes = _gpu_memory_bytes_from_registry(name) or 0.0
    memory_gb = max(adapter_ram_bytes, registry_bytes) / 2**30
    return GpuProfile(
        vendor=_gpu_vendor(name, str(entry.get("AdapterCompatibility") or "")),
        model=name,
        memory_gb=max(memory_gb, 0.0),
        available_memory_gb=None,
    )


def _gpu_rank(gpu: GpuProfile) -> tuple[int, float]:
    """Rank adapters: discrete vendors first, then by reported memory."""
    discrete = 1 if gpu.vendor in {"NVIDIA", "AMD"} else 0
    return discrete, gpu.memory_gb


def _gpu_memory_bytes_from_registry(name: str) -> float | None:
    """Read adapter memory from the display-class registry key (not uint32-capped).

    ``Win32_VideoController.AdapterRAM`` is a 32-bit field and saturates near 4 GB.
    The driver publishes the true size as ``HardwareInformation.qwMemorySize``.
    Read-only; returns ``None`` if the value is unavailable.
    """
    try:
        import winreg
    except ImportError:
        return None
    target = name.strip().lower()
    try:
        class_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY)
    except OSError:
        return None
    best: float | None = None
    try:
        for index in range(64):
            try:
                subkey_name = winreg.EnumKey(class_key, index)
            except OSError:
                break
            if not subkey_name.isdigit():
                continue
            value = _read_adapter_memory(class_key, subkey_name, target)
            if value is not None and (best is None or value > best):
                best = value
    finally:
        class_key.Close()
    return best


def _read_adapter_memory(class_key: object, subkey_name: str, target: str) -> float | None:
    """Return adapter memory bytes for one display-class subkey if its name matches."""
    import winreg

    try:
        subkey = winreg.OpenKey(class_key, subkey_name)  # type: ignore[arg-type]
    except OSError:
        return None
    try:
        try:
            desc = str(winreg.QueryValueEx(subkey, "DriverDesc")[0]).strip().lower()
        except OSError:
            desc = ""
        if target and desc and target not in desc and desc not in target:
            return None
        for value_name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
            try:
                data, _ = winreg.QueryValueEx(subkey, value_name)
            except OSError:
                continue
            if isinstance(data, int) and data > 0:
                return float(data)
            if isinstance(data, bytes) and data:
                return float(int.from_bytes(data, "little"))
    finally:
        subkey.Close()
    return None


def _gpu_vendor(name: str, adapter_compatibility: str) -> str:
    """Normalize GPU vendor from local adapter metadata."""
    source = f"{name} {adapter_compatibility}".lower()
    if "nvidia" in source:
        return "NVIDIA"
    if "amd" in source or "radeon" in source:
        return "AMD"
    if "intel" in source:
        return "Intel"
    return adapter_compatibility or "Unknown"
