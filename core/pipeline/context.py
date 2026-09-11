"""Execution context passed to every stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.editions import Entitlements
from schemas.pipeline import Artifact, ArtifactKind, StageName

if TYPE_CHECKING:
    from core.pipeline.runs import RunHandle
    from core.project import ProjectConfig


@dataclass
class RunContext:
    """Everything a stage needs: the project, its run directory, and entitlements.

    ``run`` is ``None`` during planning. ``Stage.plan`` must therefore not touch the
    run; only ``Stage.run`` may call :attr:`workspace`, :meth:`log`, or
    :meth:`input_artifact`.
    """

    project: ProjectConfig
    entitlements: Entitlements
    run: RunHandle | None = None
    params: dict[str, object] = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        """Return the stage-writable directory for this run."""
        return self._require_run().artifacts_dir

    def input_artifact(self, *, stage: StageName) -> Artifact | None:
        """Return the most recent artifact produced by an earlier stage, if any."""
        for artifact in reversed(self._require_run().record.artifacts):
            if artifact.produced_by_stage == stage:
                return artifact
        return None

    def input_artifacts(self, *, stage: StageName) -> list[Artifact]:
        """Return every artifact produced by an earlier stage, most recent first.

        Useful when a stage produces more than one artifact of the same kind (for
        example ``data-ingest``'s optional train/val split) and a caller needs to
        pick one by its ``provenance``.
        """
        return [
            artifact
            for artifact in reversed(self._require_run().record.artifacts)
            if artifact.produced_by_stage == stage
        ]

    def log(self, message: str, *, stage: StageName | None = None, level: str = "info") -> None:
        """Append an event to the run's append-only log."""
        self._require_run().append_event(message, stage=stage, level=level)

    def record_artifact(
        self,
        path: Path,
        *,
        kind: ArtifactKind,
        stage: StageName,
        provenance: dict[str, object] | None = None,
    ) -> Artifact:
        """Register a stage output with the active run."""
        return self._require_run().record_artifact(
            path, kind=kind, stage=stage, provenance=provenance
        )

    def _require_run(self) -> RunHandle:
        if self.run is None:
            raise RuntimeError("This operation is unavailable during planning (no active run).")
        return self.run
