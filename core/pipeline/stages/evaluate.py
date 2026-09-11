"""Held-out evaluation: perplexity plus a next-token top-1 accuracy probe.

Runs in-process (unlike ``pretrain``/``quantize``): a single forward pass with no
gradient step over the val split is fast even for the free tier's largest (100M)
models, so it doesn't need worker isolation. ``torch`` is imported lazily inside
``run()``, same as every other stage that touches the training framework.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.pipeline.stages._dataset import select_dataset_artifact
from core.pipeline.stages._params import int_param
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult

_DEFAULT_BLOCK_SIZE_CAP = 128
_MAX_LOSS_FOR_PERPLEXITY = 20.0  # exp(20) already astronomical; caps overflow on a fresh model


class EvaluateStage(Stage):
    """Perplexity and top-1 next-token accuracy on the held-out val split."""

    name = "evaluate"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.CHECKPOINT],
            outputs=[ArtifactKind.EVAL_REPORT],
            estimated_memory_gb=0.3,
            notes=["Held-out perplexity plus next-token top-1 accuracy on the val split."],
        )

    def run(self, ctx: RunContext) -> StageResult:
        checkpoint = ctx.input_artifact(stage="pretrain")
        if checkpoint is None:
            raise StageError("evaluate requires a 'checkpoint' artifact; run pretrain first.")
        val_dataset = select_dataset_artifact(ctx, role="val")
        if val_dataset is None:
            raise StageError(
                "evaluate requires a val split; re-run data-ingest with --val-ratio > 0."
            )
        tokenizer = ctx.input_artifact(stage="tokenizer-train")
        if tokenizer is None:
            raise StageError(
                "evaluate requires a 'tokenizer' artifact; run tokenizer-train first."
            )

        import torch

        from core.training.data import tokenize_to_blocks
        from core.training.model import LlamaConfig, LlamaModel

        ckpt = torch.load(Path(checkpoint.path), map_location="cpu", weights_only=False)
        arch = LlamaConfig(**ckpt["config"])
        model = LlamaModel(arch)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        default_block_size = min(arch.context_length - 1, _DEFAULT_BLOCK_SIZE_CAP)
        block_size = int_param(ctx.params, "block_size", default_block_size)
        blocks = tokenize_to_blocks(Path(val_dataset.path), Path(tokenizer.path), block_size)
        if not blocks:
            raise StageError(
                f"val split has fewer than {block_size + 1} tokens; lower --block-size or add data."
            )

        total_loss = 0.0
        total_correct = 0
        total_targets = 0
        with torch.no_grad():
            for block in blocks:
                inputs = block[:-1].unsqueeze(0)
                targets = block[1:].unsqueeze(0)
                logits, loss = model(inputs, labels=targets)
                assert loss is not None
                total_loss += loss.item() * targets.numel()
                total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
                total_targets += targets.numel()

        mean_loss = total_loss / total_targets
        perplexity = math.exp(min(mean_loss, _MAX_LOSS_FOR_PERPLEXITY))
        top1_accuracy = total_correct / total_targets

        report = {
            "loss": mean_loss,
            "perplexity": perplexity,
            "top1_accuracy": top1_accuracy,
            "val_blocks": len(blocks),
            "block_size": block_size,
            "checkpoint_artifact_id": checkpoint.id,
        }
        report_path = safe_resolve(ctx.workspace, "eval_report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        artifact = ctx.record_artifact(
            report_path, kind=ArtifactKind.EVAL_REPORT, stage=self.name, provenance=report
        )
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[artifact],
            metrics={"loss": mean_loss, "perplexity": perplexity, "top1_accuracy": top1_accuracy},
            message=f"perplexity {perplexity:.2f}, top-1 accuracy {top1_accuracy:.2%}.",
        )
