from core.hardware.profiler import HardwareProfiler
from core.sizing.engine import ModelFitEngine

_VALID_STATUS = {"recommended", "possible", "constrained", "not_recommended"}


def test_real_hardware_flows_into_recommendations() -> None:
    profile = HardwareProfiler().profile()
    results = ModelFitEngine().recommend(profile)
    assert len(results) == 3
    for result in results:
        assert result.status in _VALID_STATUS
        assert result.memory_required_gb > 0
        assert result.disk_required_gb > 0
        assert result.disk_available_gb >= 0
        assert result.expected_available_memory_gb >= 0
