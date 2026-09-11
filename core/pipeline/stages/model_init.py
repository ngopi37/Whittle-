"""Model architecture instantiation.

Builds the Llama-style architecture (``core.training.model``) from the project's
``ModelConfig`` and the trained tokenizer's actual vocabulary, and saves randomly
initialized weights as the ``model_init`` artifact that ``pretrain`` starts from.

This runs in-process (unlike ``pretrain``/``quantize``): instantiating an untrained
model is fast even at 500M parameters, so it doesn't need worker isolation. ``torch``
is imported lazily inside ``run()`` so importing this module — which every CLI/desktop
invocation does via the pipeline registry — never pays torch's import cost.
"""

from __future__ import annotations

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult


class ModelInitStage(Stage):
    """Instantiate the architecture and save its initial weights."""

    name = "model-init"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.TOKENIZER],
            outputs=[ArtifactKind.MODEL_INIT],
            estimated_memory_gb=round(0.2 + ctx.project.model.target_parameters * 4 / 2**30, 3),
            estimated_disk_gb=round(ctx.project.model.target_parameters * 4 / 2**30, 3),
            notes=["Llama-style decoder-only transformer; random init, fp32."],
        )

    def run(self, ctx: RunContext) -> StageResult:
        import torch

        from core.training.model import LlamaModel, config_from_model

        tokenizer = ctx.input_artifact(stage="tokenizer-train")
        if tokenizer is None:
            raise StageError(
                "model-init requires a 'tokenizer' artifact; run tokenizer-train first."
            )
        vocab_size = int(tokenizer.provenance.get("vocab_size_actual", 0))
        if vocab_size <= 0:
            raise StageError("tokenizer artifact is missing a valid 'vocab_size_actual'.")

        arch = config_from_model(ctx.project.model, vocab_size=vocab_size)
        model = LlamaModel(arch)

        model_path = safe_resolve(ctx.workspace, "model_init.pt")
        torch.save(
            {
                "schema_version": 1,
                "state_dict": model.state_dict(),
                "config": arch.__dict__,
                "tokenizer_artifact_id": tokenizer.id,
            },
            model_path,
        )
        ctx.log(
            f"Initialized {model.num_parameters():,} parameter Llama-style model "
            f"(vocab {vocab_size}).",
            stage=self.name,
        )
        artifact = ctx.record_artifact(
            model_path,
            kind=ArtifactKind.MODEL_INIT,
            stage=self.name,
            provenance={
                "parameters": model.num_parameters(),
                "vocab_size": vocab_size,
                "hidden_size": arch.hidden_size,
                "layers": arch.layers,
                "attention_heads": arch.attention_heads,
                "intermediate_size": arch.intermediate_size,
                "context_length": arch.context_length,
                "tokenizer_artifact_id": tokenizer.id,
            },
        )
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[artifact],
            metrics={"parameters": float(model.num_parameters())},
            message=f"Initialized {model.num_parameters():,} parameters.",
        )
