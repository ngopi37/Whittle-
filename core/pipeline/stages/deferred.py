"""Placeholder stages that plan but do not run yet."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.pipeline.base import Stage, StageNotImplemented
from core.pipeline.context import RunContext
from schemas.pipeline import ArtifactKind, StageName, StagePlan, StageResult


@dataclass
class DeferredStage(Stage):
    """A stage declared in the canonical pipeline but implemented in a later phase.

    ``plan`` is real so the whole build is visible in ``pipeline plan``; ``run``
    raises :class:`StageNotImplemented` pointing at the roadmap phase.
    """

    name: StageName
    roadmap_phase: str
    required_feature: str | None = None
    inputs: list[ArtifactKind] = field(default_factory=list)
    outputs: list[ArtifactKind] = field(default_factory=list)
    estimated_memory_gb: float = 0.0
    estimated_disk_gb: float = 0.0
    notes: list[str] = field(default_factory=list)

    def plan(self, ctx: RunContext) -> StagePlan:
        notes = [f"Deferred to roadmap {self.roadmap_phase}.", *self.notes]
        return StagePlan(
            stage=self.name,
            implemented=False,
            required_feature=self.required_feature,
            inputs=self.inputs,
            outputs=self.outputs,
            estimated_memory_gb=self.estimated_memory_gb,
            estimated_disk_gb=self.estimated_disk_gb,
            notes=notes,
        )

    def run(self, ctx: RunContext) -> StageResult:
        raise StageNotImplemented(self.name, self.roadmap_phase)
