"""Pipeline, run, artifact, and job data contracts.

These types are the stable boundary between the orchestrator, the stage
implementations, and the (future) out-of-process workers. Bump ``JobSpec.schema_version``
on any breaking change to the job protocol.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StageName = Literal[
    "data-ingest",
    "tokenizer-train",
    "model-init",
    "pretrain",
    "evaluate",
    "quantize",
    "package",
    "on-device-smoke-test",
]

#: Canonical stage order for the on-device model build.
STAGE_ORDER: tuple[StageName, ...] = (
    "data-ingest",
    "tokenizer-train",
    "model-init",
    "pretrain",
    "evaluate",
    "quantize",
    "package",
    "on-device-smoke-test",
)

RunStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


class ArtifactKind(StrEnum):
    DATASET = "dataset"
    TOKENIZER = "tokenizer"
    MODEL_INIT = "model_init"
    CHECKPOINT = "checkpoint"
    EVAL_REPORT = "eval_report"
    QUANTIZED_MODEL = "quantized_model"
    PACKAGE = "package"


class Artifact(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    kind: ArtifactKind
    path: str
    sha256: str
    bytes: int = Field(ge=0)
    produced_by_stage: StageName
    provenance: dict[str, Any] = Field(default_factory=dict)


class StagePlan(BaseModel):
    """A stage's declared inputs, outputs, cost, and gating — computed without running."""

    model_config = ConfigDict(protected_namespaces=())

    stage: StageName
    implemented: bool
    required_feature: str | None = None
    inputs: list[ArtifactKind] = Field(default_factory=list)
    outputs: list[ArtifactKind] = Field(default_factory=list)
    estimated_memory_gb: float = Field(default=0.0, ge=0)
    estimated_disk_gb: float = Field(default=0.0, ge=0)
    notes: list[str] = Field(default_factory=list)


class StageResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    stage: StageName
    status: RunStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = ""


class PipelinePlan(BaseModel):
    project_name: str
    model_size: str
    stages: list[StagePlan]
    total_estimated_memory_gb: float = Field(ge=0)
    total_estimated_disk_gb: float = Field(ge=0)


class RunEvent(BaseModel):
    at: datetime
    stage: StageName | None = None
    level: Literal["info", "warning", "error"] = "info"
    message: str


class RunRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    project_name: str
    model_size: str
    created_at: datetime
    status: RunStatus
    stages: dict[str, RunStatus] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)


class JobSpec(BaseModel):
    """Versioned unit of work handed to an out-of-process worker."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal[1] = 1
    job_type: Literal["stage"] = "stage"
    stage: StageName
    project_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, float] = Field(default_factory=dict)


class JobEvent(BaseModel):
    schema_version: Literal[1] = 1
    at: datetime
    kind: Literal["progress", "metric", "log", "artifact", "done", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)
