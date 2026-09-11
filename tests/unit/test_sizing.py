from core.sizing.engine import SAFETY_FRACTION, ModelFitEngine
from schemas.hardware import CpuProfile, GpuProfile, HardwareProfile, MemoryProfile, StorageProfile
from schemas.model import TrainingConfig


def fixture_hardware(
    available_gb: float = 12, free_disk_gb: float = 100, gpu: GpuProfile | None = None
) -> HardwareProfile:
    return HardwareProfile(
        os="Windows",
        architecture="x86_64",
        cpu=CpuProfile(vendor="Intel", model="Intel i7-10750H", physical_cores=6, logical_cores=12),
        memory=MemoryProfile(total_gb=16, available_gb=available_gb),
        gpu=gpu or GpuProfile(),
        storage=StorageProfile(capacity_gb=500, free_gb=free_disk_gb),
    )


def test_i7_16gb_cpu_only_recommendation_is_ordered() -> None:
    results = {item.model_size: item for item in ModelFitEngine().recommend(fixture_hardware())}
    fifty = results["50M"].resources
    assert results["20M"].status == "recommended"
    assert fifty.optimizer_memory_gb > fifty.gradient_memory_gb
    assert fifty.estimated_compute_difficulty == "medium"
    # Peak-RAM estimate rises monotonically with model size.
    assert (
        results["20M"].memory_required_gb
        < results["50M"].memory_required_gb
        < results["100M"].memory_required_gb
    )


def test_low_memory_is_not_recommended() -> None:
    result = ModelFitEngine().evaluate(fixture_hardware(available_gb=1), "20M")
    assert result.status == "not_recommended"
    assert result.reasons


def test_training_config_changes_the_estimate() -> None:
    hardware = fixture_hardware()
    baseline = ModelFitEngine().evaluate(hardware, "50M")
    heavier = ModelFitEngine().evaluate(
        hardware, "50M", training=TrainingConfig(batch_size=16)
    )
    assert heavier.memory_required_gb > baseline.memory_required_gb
    assert heavier.resources.activation_memory_gb > baseline.resources.activation_memory_gb


def test_mixed_precision_reduces_activation_memory() -> None:
    hardware = fixture_hardware()
    full = ModelFitEngine().evaluate(hardware, "100M", training=TrainingConfig(batch_size=8))
    mixed = ModelFitEngine().evaluate(
        hardware, "100M", training=TrainingConfig(batch_size=8, mixed_precision=True)
    )
    assert mixed.resources.activation_memory_gb < full.resources.activation_memory_gb
    assert mixed.resources.master_copy_memory_gb > 0


def test_dataset_storage_is_not_counted_as_ram() -> None:
    result = ModelFitEngine().evaluate(fixture_hardware(), "50M")
    assert result.resources.dataset_storage_gb > 0
    # memory_required is the component subtotal plus the safety margin only.
    expected = round(
        result.resources.component_subtotal_gb + result.resources.safety_margin_gb, 2
    )
    assert result.memory_required_gb == expected


def test_insufficient_disk_caps_the_status() -> None:
    plenty_ram_no_disk = fixture_hardware(available_gb=64, free_disk_gb=0.4)
    result = ModelFitEngine().evaluate(plenty_ram_no_disk, "100M")
    assert result.status == "not_recommended"
    assert any("disk" in reason.lower() for reason in result.reasons)


def test_engine_can_evaluate_future_candidate_sizes() -> None:
    result = ModelFitEngine().evaluate(fixture_hardware(available_gb=64), "500M")
    assert result.model_size == "500M"
    assert result.status in {"possible", "constrained", "not_recommended", "recommended"}
    assert result.estimated_training_time.min_hours < result.estimated_training_time.max_hours


def test_safety_margin_is_a_transparent_fraction() -> None:
    result = ModelFitEngine().evaluate(fixture_hardware(), "20M")
    assert result.resources.safety_margin_gb == round(
        SAFETY_FRACTION * result.resources.component_subtotal_gb, 2
    )
