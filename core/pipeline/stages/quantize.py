"""GGUF quantization — the only runtime target implemented so far.

ExecuTorch and TFLite/Core ML are still ``DeferredStage`` territory: see
``docs/runtime-targets.md`` for why (different, heavy toolchains; Core ML in
particular can't be verified off macOS). This stage runs in-process — unlike
``pretrain``, exporting/quantizing a free-tier (≤100M) checkpoint is a handful of
numpy-vectorized tensor writes, not a long training loop, so it doesn't need worker
isolation. ``torch``/``gguf`` are imported lazily inside ``run()``.
"""

from __future__ import annotations

from pathlib import Path

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult

_QUANT_CHOICES = ("fp16", "q8_0", "q4_0")
_DEFAULT_QUANT = "q8_0"


class QuantizeStage(Stage):
    """Export the trained checkpoint to a quantized GGUF file."""

    name = "quantize"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.CHECKPOINT],
            outputs=[ArtifactKind.QUANTIZED_MODEL],
            notes=[
                "GGUF/llama.cpp only for now; quant types: fp16, q8_0 (default), q4_0.",
                "ExecuTorch and TFLite/Core ML remain deferred — see docs/runtime-targets.md.",
            ],
        )

    def run(self, ctx: RunContext) -> StageResult:
        checkpoint = ctx.input_artifact(stage="pretrain")
        if checkpoint is None:
            raise StageError("quantize requires a 'checkpoint' artifact; run pretrain first.")
        tokenizer = ctx.input_artifact(stage="tokenizer-train")
        if tokenizer is None:
            raise StageError("quantize requires a 'tokenizer' artifact; run tokenizer-train first.")

        quant_type = str(ctx.params.get("quant_type") or _DEFAULT_QUANT).lower()
        if quant_type not in _QUANT_CHOICES:
            raise StageError(
                f"Unknown quant_type {quant_type!r}; expected one of {_QUANT_CHOICES}."
            )

        from core.runtime.gguf_export import export_gguf

        out_path = safe_resolve(ctx.workspace, f"model.{quant_type}.gguf")
        stats = export_gguf(
            checkpoint_path=Path(checkpoint.path),
            tokenizer_path=Path(tokenizer.path),
            out_path=out_path,
            quant_type=quant_type,
        )
        ctx.log(
            f"Exported GGUF ({quant_type}): {stats['tensors']} tensors, {stats['bytes']:,} bytes.",
            stage=self.name,
        )
        artifact = ctx.record_artifact(
            out_path,
            kind=ArtifactKind.QUANTIZED_MODEL,
            stage=self.name,
            provenance={
                "runtime": "gguf",
                "quant_type": quant_type,
                "tensors": stats["tensors"],
                "vocab_size": stats["vocab_size"],
                "layers": stats["layers"],
                "checkpoint_artifact_id": checkpoint.id,
                "tokenizer_artifact_id": tokenizer.id,
            },
        )
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[artifact],
            metrics={"bytes": float(stats["bytes"]), "tensors": float(stats["tensors"])},
            message=f"Quantized to GGUF {quant_type}, {stats['bytes']:,} bytes.",
        )
