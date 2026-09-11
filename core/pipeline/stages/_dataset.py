"""Shared helper for picking a dataset artifact by split role.

``data-ingest`` may record one artifact (``role: "all"``, no split requested) or
two (``role: "train"`` and ``role: "val"``, when ``val_ratio`` was given). Every
downstream stage that consumes a dataset goes through :func:`select_dataset_artifact`
so the role-selection logic lives in exactly one place.
"""

from __future__ import annotations

from core.pipeline.context import RunContext
from schemas.pipeline import Artifact, StageName

_INGEST_STAGE: StageName = "data-ingest"


def select_dataset_artifact(ctx: RunContext, *, role: str) -> Artifact | None:
    """Return the data-ingest artifact matching ``role`` ("train" or "val").

    ``"train"`` also matches an unsplit dataset (``role: "all"``), so stages that
    just want "the training data" work whether or not a val split was requested.
    """
    for artifact in ctx.input_artifacts(stage=_INGEST_STAGE):
        artifact_role = artifact.provenance.get("role", "all")
        if role == "train" and artifact_role in ("train", "all"):
            return artifact
        if role == "val" and artifact_role == "val":
            return artifact
    return None
