"""Hardware-aware model fit calculations.

Every figure here is a conservative *planning* estimate expressed as an explicit,
named function of measured hardware plus the project's model and training config.
There is no trainer yet to calibrate against, so the numbers lean pessimistic and
durations are reported as ranges.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.catalog import MODEL_PARAMETERS, USER_FACING_SIZES, get_spec
from schemas.hardware import HardwareProfile
from schemas.model import ModelConfig, TrainingConfig

Status = Literal["recommended", "possible", "constrained", "not_recommended"]

#: Fraction of the component subtotal reserved as headroom for fragmentation,
#: framework overhead, and measurement error.
SAFETY_FRACTION = 0.25

#: Checkpoint copies kept on disk (current + best + rolling backup).
CHECKPOINT_COPIES = 2.5

_BYTES_PER_GB = 2**30

#: Backwards-compatible mapping used by older call sites and tests.
MODEL_SIZES: dict[str, int] = dict(MODEL_PARAMETERS)


class TimeRange(BaseModel):
    min_hours: float = Field(ge=0)
    max_hours: float = Field(ge=0)


class ResourceEstimate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    estimated_model_weight_memory_gb: float = Field(ge=0)
    master_copy_memory_gb: float = Field(ge=0)
    optimizer_memory_gb: float = Field(ge=0)
    gradient_memory_gb: float = Field(ge=0)
    activation_memory_gb: float = Field(ge=0)
    tokenizer_memory_gb: float = Field(ge=0)
    data_pipeline_memory_gb: float = Field(ge=0)
    runtime_overhead_gb: float = Field(ge=0)
    checkpoint_storage_gb: float = Field(ge=0)
    dataset_storage_gb: float = Field(ge=0)
    safety_margin_gb: float = Field(ge=0)
    expected_available_memory_gb: float = Field(ge=0)
    estimated_compute_difficulty: Literal["low", "medium", "high", "very_high"]

    @property
    def component_subtotal_gb(self) -> float:
        """Return the sum of real RAM components, before the safety margin."""
        return round(
            self.estimated_model_weight_memory_gb
            + self.master_copy_memory_gb
            + self.optimizer_memory_gb
            + self.gradient_memory_gb
            + self.activation_memory_gb
            + self.tokenizer_memory_gb
            + self.data_pipeline_memory_gb
            + self.runtime_overhead_gb,
            2,
        )

    @property
    def memory_required_gb(self) -> float:
        """Return total estimated peak RAM (components plus safety margin).

        Dataset and checkpoint figures are *disk*, not RAM, and are excluded here.
        """
        return round(self.component_subtotal_gb + self.safety_margin_gb, 2)

    @property
    def disk_required_gb(self) -> float:
        """Return total estimated local disk for checkpoints plus dataset storage."""
        return round(self.checkpoint_storage_gb + self.dataset_storage_gb, 2)


class FitResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_size: str
    parameters: int
    status: Status
    confidence: Literal["low", "medium", "high"]
    memory_required_gb: float
    expected_available_memory_gb: float
    disk_required_gb: float
    disk_available_gb: float
    resources: ResourceEstimate
    checkpoint_storage_gb: float
    estimated_training_time: TimeRange
    reasons: list[str]


class ModelFitEngine:
    """Estimate training fit using measured resources and conservative overheads."""

    def evaluate(
        self,
        hardware: HardwareProfile,
        model_size: str,
        *,
        training: TrainingConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> FitResult:
        """Classify one candidate size and explain every resource decision."""
        spec = get_spec(model_size)
        parameters = spec.parameters
        training = training or TrainingConfig()
        model_config = model_config or _default_architecture(model_size, parameters)

        resources = self._estimate_resources(hardware, parameters, training, model_config)
        memory_required = resources.memory_required_gb
        available = hardware.memory.available_gb
        ratio = available / memory_required if memory_required else 0.0

        status, confidence, reason = _classify_memory(ratio, hardware)

        disk_required = resources.disk_required_gb
        disk_available = round(hardware.storage.free_gb, 2)
        status, disk_reason = _apply_disk_gating(status, disk_required, disk_available)

        hours = self._estimate_duration_hours(hardware, parameters, resources, training)
        reasons = [
            reason,
            f"Estimated peak training memory is {memory_required:.1f} GB "
            f"({resources.component_subtotal_gb:.1f} GB components + "
            f"{resources.safety_margin_gb:.1f} GB safety margin).",
            f"The device reports {available:.1f} GB available RAM.",
            f"Local disk needed is {disk_required:.1f} GB "
            f"(checkpoints {resources.checkpoint_storage_gb:.1f} GB + "
            f"dataset {resources.dataset_storage_gb:.1f} GB); "
            f"{disk_available:.1f} GB free.",
            f"Compute difficulty is {resources.estimated_compute_difficulty.replace('_', ' ')}.",
            f"Assumes batch size {training.batch_size}, context "
            f"{model_config.context_length}, "
            f"{'mixed' if training.mixed_precision else 'full'}-precision.",
        ]
        if disk_reason:
            reasons.append(disk_reason)
        if hardware.gpu.vendor == "None":
            reasons.append("CPU-only execution is supported, but training will be slower.")
        else:
            reasons.append(f"Detected {hardware.gpu.vendor} GPU: {hardware.gpu.model}.")

        return FitResult(
            model_size=model_size,
            parameters=parameters,
            status=status,
            confidence=confidence,
            memory_required_gb=round(memory_required, 2),
            expected_available_memory_gb=round(available, 2),
            disk_required_gb=disk_required,
            disk_available_gb=disk_available,
            resources=resources,
            checkpoint_storage_gb=resources.checkpoint_storage_gb,
            estimated_training_time=TimeRange(
                min_hours=round(hours, 1), max_hours=round(hours * 2.5, 1)
            ),
            reasons=reasons,
        )

    def recommend(
        self, hardware: HardwareProfile, training: TrainingConfig | None = None
    ) -> list[FitResult]:
        """Evaluate the user-facing candidate sizes in ascending order."""
        return [self.evaluate(hardware, size, training=training) for size in USER_FACING_SIZES]

    def _estimate_resources(
        self,
        hardware: HardwareProfile,
        parameters: int,
        training: TrainingConfig,
        model_config: ModelConfig,
    ) -> ResourceEstimate:
        """Calculate explicit memory, storage, and compute components."""
        mp = training.mixed_precision
        act_bytes = 2 if mp else 4

        weights = parameters * 4 / _BYTES_PER_GB
        master_copy = (parameters * 2 / _BYTES_PER_GB) if mp else 0.0
        gradients = parameters * (2 if mp else 4) / _BYTES_PER_GB
        optimizer = parameters * 8 / _BYTES_PER_GB  # Adam moments m, v in fp32

        tokens_in_flight = training.batch_size * model_config.context_length
        feature_acts = (
            tokens_in_flight
            * model_config.hidden_size
            * model_config.layers
            * (2 + model_config.ffn_multiplier)
            * act_bytes
        )
        attention_acts = (
            training.batch_size
            * model_config.attention_heads
            * model_config.context_length**2
            * model_config.layers
            * act_bytes
        )
        activations = (feature_acts + attention_acts) / _BYTES_PER_GB

        tokenizer = 0.05 + model_config.vocabulary_size * 8e-7
        data_pipeline = 0.2 + training.batch_size * 0.01
        # Python, the training framework, and dataloader workers; plus an accelerator
        # allocator/context reservation when a GPU is present.
        runtime_overhead = 0.5 + (1.0 if hardware.gpu.vendor != "None" else 0.0)

        parameter_millions = parameters / 1_000_000
        dataset_storage = 0.35 + parameter_millions * 0.004
        checkpoint_storage = weights * CHECKPOINT_COPIES

        weights = round(weights, 2)
        master_copy = round(master_copy, 2)
        gradients = round(gradients, 2)
        optimizer = round(optimizer, 2)
        activations = round(activations, 2)
        tokenizer = round(tokenizer, 2)
        data_pipeline = round(data_pipeline, 2)
        runtime_overhead = round(runtime_overhead, 2)

        subtotal = (
            weights
            + master_copy
            + gradients
            + optimizer
            + activations
            + tokenizer
            + data_pipeline
            + runtime_overhead
        )
        safety = round(SAFETY_FRACTION * subtotal, 2)

        return ResourceEstimate(
            estimated_model_weight_memory_gb=weights,
            master_copy_memory_gb=master_copy,
            optimizer_memory_gb=optimizer,
            gradient_memory_gb=gradients,
            activation_memory_gb=activations,
            tokenizer_memory_gb=tokenizer,
            data_pipeline_memory_gb=data_pipeline,
            runtime_overhead_gb=runtime_overhead,
            checkpoint_storage_gb=round(checkpoint_storage, 2),
            dataset_storage_gb=round(dataset_storage, 2),
            safety_margin_gb=safety,
            expected_available_memory_gb=round(hardware.memory.available_gb, 2),
            estimated_compute_difficulty=_compute_difficulty(hardware, parameters),
        )

    def _estimate_duration_hours(
        self,
        hardware: HardwareProfile,
        parameters: int,
        resources: ResourceEstimate,
        training: TrainingConfig,
    ) -> float:
        """Estimate a broad duration floor from CPU/GPU capacity indicators."""
        cpu_factor = max(hardware.cpu.logical_cores / 4, 0.5)
        if (
            hardware.gpu.vendor == "NVIDIA"
            and hardware.gpu.memory_gb >= resources.memory_required_gb * 0.6
        ):
            cpu_factor *= 2.5
        elif hardware.gpu.vendor != "None":
            cpu_factor *= 1.4
        difficulty_factor = {"low": 0.8, "medium": 1.0, "high": 1.35, "very_high": 1.8}[
            resources.estimated_compute_difficulty
        ]
        effective_batch = training.batch_size * training.gradient_accumulation_steps
        work_factor = training.epochs * max(effective_batch / 8, 0.5)
        base = parameters / 10_000_000 / cpu_factor * 2.5 * difficulty_factor
        return max(base * work_factor, 0.5)


def _default_architecture(model_size: str, parameters: int) -> ModelConfig:
    """Return a minimal architecture profile when the caller supplies none."""
    return ModelConfig(
        name=model_size,
        target_parameters=parameters,
        vocabulary_size=32_000,
        context_length=1024,
        layers=max(round(parameters / 8_000_000), 4),
        hidden_size=512,
        attention_heads=8,
        ffn_multiplier=4.0,
    )


def _classify_memory(
    ratio: float, hardware: HardwareProfile
) -> tuple[Status, Literal["low", "medium", "high"], str]:
    """Classify status and confidence from the available/required memory ratio."""
    if ratio >= 2.0:
        confidence: Literal["low", "medium", "high"] = (
            "medium" if hardware.gpu.vendor == "None" else "high"
        )
        return "recommended", confidence, (
            "Measured available memory provides at least a 2x safety margin."
        )
    if ratio >= 1.25:
        return "possible", "medium", (
            "The workload fits the measured memory, but headroom is limited."
        )
    if ratio >= 0.9:
        return "constrained", "low", (
            "The estimate is near the available-memory limit; reduce batch size."
        )
    return "not_recommended", "high", (
        "Estimated training memory exceeds measured available memory."
    )


_STATUS_ORDER: tuple[Status, ...] = (
    "recommended",
    "possible",
    "constrained",
    "not_recommended",
)


def _apply_disk_gating(
    status: Status, disk_required: float, disk_available: float
) -> tuple[Status, str | None]:
    """Cap status when local disk is short of the checkpoint and dataset estimate."""
    if disk_available <= 0 or disk_required <= 0:
        return status, None
    if disk_available < disk_required * 0.5:
        return "not_recommended", (
            f"Free disk ({disk_available:.1f} GB) is far below the "
            f"{disk_required:.1f} GB needed for checkpoints and dataset storage."
        )
    if disk_available < disk_required:
        capped = _STATUS_ORDER[max(_STATUS_ORDER.index(status), _STATUS_ORDER.index("constrained"))]
        return capped, (
            f"Free disk ({disk_available:.1f} GB) is below the estimated "
            f"{disk_required:.1f} GB; free space or relocate the workspace."
        )
    return status, None


def _compute_difficulty(
    hardware: HardwareProfile, parameters: int
) -> Literal["low", "medium", "high", "very_high"]:
    """Classify compute difficulty from model size, cores, and GPU metadata."""
    if parameters <= 20_000_000 and hardware.cpu.logical_cores >= 8:
        return "low"
    if parameters <= 50_000_000 and (
        hardware.cpu.logical_cores >= 12 or hardware.gpu.vendor != "None"
    ):
        return "medium"
    if parameters <= 100_000_000 and hardware.cpu.logical_cores >= 8:
        return "high"
    return "very_high"
