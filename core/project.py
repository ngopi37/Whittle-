"""Local project configuration persistence."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.models import ModelConfigGenerator
from core.sizing.engine import FitResult, ModelFitEngine
from schemas.hardware import HardwareProfile
from schemas.model import ModelConfig, TrainingConfig

__all__ = ["ProjectConfig", "TrainingConfig", "create_project", "refresh_recommendation"]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareProfile | None = None
    recommendation: FitResult | None = None

    def save(self, path: Path) -> None:
        """Write a project configuration as local JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ProjectConfig:
        """Load and validate a project configuration from JSON."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def create_project(
    name: str,
    model_size: str,
    hardware: HardwareProfile | None = None,
    recommendation: FitResult | None = None,
    training: TrainingConfig | None = None,
) -> ProjectConfig:
    """Create a project from an initial model profile."""
    return ProjectConfig(
        name=name,
        model=ModelConfigGenerator().generate(model_size),
        training=training or TrainingConfig(),
        hardware=hardware,
        recommendation=recommendation,
    )


def refresh_recommendation(project: ProjectConfig, hardware: HardwareProfile) -> ProjectConfig:
    """Return a copy of the project with hardware and recommendation recomputed.

    The recommendation is evaluated against the project's own training and model
    configuration, so it stays meaningful as those knobs change.
    """
    recommendation = ModelFitEngine().evaluate(
        hardware,
        project.model.name,
        training=project.training,
        model_config=project.model,
    )
    return project.model_copy(update={"hardware": hardware, "recommendation": recommendation})
