from pathlib import Path

from core.project import ProjectConfig, create_project
from core.sizing.engine import ModelFitEngine
from schemas.hardware import CpuProfile, GpuProfile, HardwareProfile, MemoryProfile, StorageProfile


def fixture_hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Windows",
        architecture="x86_64",
        cpu=CpuProfile(vendor="Intel", model="Intel i7-10750H", physical_cores=6, logical_cores=12),
        memory=MemoryProfile(total_gb=16, available_gb=12),
        gpu=GpuProfile(),
        storage=StorageProfile(capacity_gb=500, free_gb=100),
    )


def test_project_can_be_saved_and_reopened(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    original = create_project("support-model", "20M")
    original.save(path)
    reopened = ProjectConfig.load(path)
    assert reopened.name == "support-model"
    assert reopened.model.name == "20M"


def test_project_can_store_phase_1_evidence(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    hardware = fixture_hardware()
    recommendation = ModelFitEngine().evaluate(hardware, "20M")
    original = create_project("support-model", "20M", hardware, recommendation)
    original.save(path)
    reopened = ProjectConfig.load(path)
    assert reopened.hardware is not None
    assert reopened.recommendation is not None
    assert reopened.recommendation.model_size == "20M"
