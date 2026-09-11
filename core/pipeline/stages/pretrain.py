"""Pretraining, run in an out-of-process worker.

This module never imports ``torch`` — it shells out to
``core.pipeline.workers.train_worker`` and reads its streamed ``JobEvent``s, exactly
as ``docs/pipeline.md`` requires for long-running stages. Checkpoints live under a
project-scoped directory (survives across runs, not just within one run's
workspace), so re-running ``pipeline run ... --stage pretrain`` after a kill resumes
from the last saved step instead of starting over.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.pipeline.stages._dataset import select_dataset_artifact
from core.pipeline.stages._params import int_param
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, JobEvent, JobSpec, StagePlan, StageResult

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MAX_STEPS = 200
_DEFAULT_BLOCK_SIZE_CAP = 128


class PretrainStage(Stage):
    """Train the initialized model on the ingested dataset via a worker subprocess."""

    name = "pretrain"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.DATASET, ArtifactKind.MODEL_INIT],
            outputs=[ArtifactKind.CHECKPOINT],
            estimated_memory_gb=round(0.2 + ctx.project.model.target_parameters * 8 / 2**30, 3),
            notes=[
                "Runs out-of-process (core.pipeline.workers.train_worker); "
                "resumable from the project checkpoint directory."
            ],
        )

    def run(self, ctx: RunContext) -> StageResult:
        dataset = select_dataset_artifact(ctx, role="train")
        model_init = ctx.input_artifact(stage="model-init")
        tokenizer = ctx.input_artifact(stage="tokenizer-train")
        if dataset is None:
            raise StageError("pretrain requires a 'dataset' artifact; run data-ingest first.")
        if model_init is None:
            raise StageError("pretrain requires a 'model_init' artifact; run model-init first.")
        if tokenizer is None:
            raise StageError("pretrain requires a 'tokenizer' artifact; run tokenizer-train first.")

        checkpoint_dir = Path.cwd() / ".sg2" / "checkpoints" / ctx.project.name / "pretrain"
        training = ctx.project.training
        default_block_size = min(ctx.project.model.context_length - 1, _DEFAULT_BLOCK_SIZE_CAP)
        job = JobSpec(
            schema_version=1,
            job_type="stage",
            stage=self.name,
            project_name=ctx.project.name,
            params={
                "dataset_path": dataset.path,
                "model_init_path": model_init.path,
                "tokenizer_path": tokenizer.path,
                "checkpoint_dir": str(checkpoint_dir),
                "block_size": int_param(ctx.params, "block_size", default_block_size),
                "batch_size": training.batch_size,
                "learning_rate": training.learning_rate,
                "gradient_accumulation_steps": training.gradient_accumulation_steps,
                "mixed_precision": training.mixed_precision,
                "max_steps": int_param(ctx.params, "max_steps", _DEFAULT_MAX_STEPS),
                "checkpoint_every": int_param(ctx.params, "checkpoint_every", 50),
                "seed": int_param(ctx.params, "seed", 1234),
            },
        )
        jobspec_path = safe_resolve(ctx.workspace, "pretrain.jobspec.json")
        jobspec_path.write_text(job.model_dump_json(), encoding="utf-8")

        max_steps = int(job.params["max_steps"])
        final = _run_worker(jobspec_path, ctx=ctx, stage_name=self.name, max_steps=max_steps)

        checkpoint_path = Path(str(final["checkpoint_path"]))
        artifact = ctx.record_artifact(
            checkpoint_path,
            kind=ArtifactKind.CHECKPOINT,
            stage=self.name,
            provenance={
                "final_loss": final["final_loss"],
                "steps": final["steps"],
                "checkpoint_dir": str(checkpoint_dir),
                "model_init_artifact_id": model_init.id,
            },
        )
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[artifact],
            metrics={"loss": float(final["final_loss"]), "steps": float(final["steps"])},
            message=f"Trained {int(final['steps'])} step(s), final loss {final['final_loss']:.4f}.",
        )


def _run_worker(
    jobspec_path: Path, *, ctx: RunContext, stage_name: Any, max_steps: int
) -> dict[str, Any]:
    """Spawn the training worker and forward its streamed events onto ``ctx``."""
    log_every = max(1, max_steps // 20)
    with subprocess.Popen(
        [sys.executable, "-m", "core.pipeline.workers.train_worker", str(jobspec_path)],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.stdout is not None
        done_payload: dict[str, Any] | None = None
        error_message: str | None = None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            event = JobEvent.model_validate_json(line)
            if event.kind == "log":
                ctx.log(str(event.payload.get("message", "")), stage=stage_name)
            elif event.kind == "metric" and int(event.payload.get("step", 0)) % log_every == 0:
                ctx.log(
                    f"step {event.payload.get('step')}: loss {event.payload.get('loss'):.4f}",
                    stage=stage_name,
                )
            elif event.kind == "done":
                done_payload = event.payload
            elif event.kind == "error":
                error_message = str(event.payload.get("message", "unknown worker error"))
        process.wait()
        stderr = process.stderr.read() if process.stderr else ""

    if error_message is not None or process.returncode != 0:
        raise StageError(
            f"pretrain worker failed: {error_message or f'exit code {process.returncode}'}"
            + (f"\n{stderr}" if stderr else "")
        )
    if done_payload is None:
        raise StageError("pretrain worker exited without reporting completion.")
    return done_payload
