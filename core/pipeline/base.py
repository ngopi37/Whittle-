"""Stage interface and pipeline errors."""

from __future__ import annotations

import abc

from core.pipeline.context import RunContext
from schemas.pipeline import StageName, StagePlan, StageResult


class StageError(RuntimeError):
    """A stage failed while running."""


class StageNotImplemented(StageError):
    """A stage is declared in the pipeline but not implemented in this phase."""

    def __init__(self, stage: StageName, roadmap_phase: str) -> None:
        self.stage = stage
        self.roadmap_phase = roadmap_phase
        super().__init__(
            f"Stage '{stage}' is not implemented yet (roadmap {roadmap_phase}). "
            f"See docs/product/roadmap.md."
        )


class Stage(abc.ABC):
    """One step of the on-device model build.

    ``plan`` must be cheap and side-effect free: it declares what the stage would
    consume, produce, and cost. ``run`` does the work and records artifacts through
    the :class:`RunContext`.
    """

    name: StageName

    @abc.abstractmethod
    def plan(self, ctx: RunContext) -> StagePlan:
        """Return the stage's declared inputs, outputs, cost, and gating."""

    @abc.abstractmethod
    def run(self, ctx: RunContext) -> StageResult:
        """Execute the stage and return its result."""
