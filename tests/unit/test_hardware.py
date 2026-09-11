from core.hardware.profiler import (
    _PROCESSOR_FEATURES,
    _gpu_rank,
    _gpu_vendor,
    _normalize_architecture,
    _select_gpu,
    _windows_architecture_name,
)
from schemas.hardware import GpuProfile


def test_architecture_normalization() -> None:
    assert _normalize_architecture("AMD64") == "x86_64"
    assert _normalize_architecture("aarch64") == "arm64"


def test_windows_architecture_codes() -> None:
    assert _windows_architecture_name(9) == "x86_64"
    assert _windows_architecture_name(12) == "arm64"
    assert _windows_architecture_name(999) is None


def test_gpu_vendor_normalization() -> None:
    assert _gpu_vendor("NVIDIA GeForce RTX", "") == "NVIDIA"
    assert _gpu_vendor("Radeon Graphics", "") == "AMD"
    assert _gpu_vendor("Iris Xe", "Intel Corporation") == "Intel"


def test_processor_feature_ids_match_winnt_constants() -> None:
    # Guards against the historical mix-up (10 is SSE2, not SSE3; AVX is 39, not 34).
    assert _PROCESSOR_FEATURES[10] == "sse2"
    assert _PROCESSOR_FEATURES[13] == "sse3"
    assert _PROCESSOR_FEATURES[39] == "avx"
    assert _PROCESSOR_FEATURES[40] == "avx2"
    assert _PROCESSOR_FEATURES[41] == "avx512f"


def test_select_gpu_prefers_discrete_adapter() -> None:
    raw = [
        {"Name": "Intel UHD Graphics 630", "AdapterRAM": 2**30, "AdapterCompatibility": "Intel"},
        {"Name": "NVIDIA GeForce RTX 3060", "AdapterRAM": 4293918720, "AdapterCompatibility": "NV"},
    ]
    selected = _select_gpu(raw)
    assert selected.vendor == "NVIDIA"


def test_select_gpu_ignores_basic_display_adapter() -> None:
    raw = [{"Name": "Microsoft Basic Display Adapter", "AdapterRAM": 0}]
    assert _select_gpu(raw) == GpuProfile()


def test_gpu_rank_orders_discrete_first_then_memory() -> None:
    intel = GpuProfile(vendor="Intel", model="Iris", memory_gb=8)
    nvidia = GpuProfile(vendor="NVIDIA", model="RTX", memory_gb=6)
    assert _gpu_rank(nvidia) > _gpu_rank(intel)
