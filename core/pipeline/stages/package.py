"""Bundle the quantized GGUF file, tokenizer, and a manifest into one archive.

llama.cpp reads the tokenizer straight out of the GGUF file (``quantize`` embeds it
— see ``core.runtime.gguf_export``), so there's no separate tokenizer file to ship;
``package`` records the tokenizer's hash in the manifest for provenance and bundles
the ``.gguf`` + ``manifest.json`` into a single ``.zip`` a device can be handed.
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult


class PackageStage(Stage):
    """Bundle the quantized model + manifest into a single distributable archive."""

    name = "package"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.QUANTIZED_MODEL, ArtifactKind.TOKENIZER],
            outputs=[ArtifactKind.PACKAGE],
            notes=["GGUF only: a .zip of the .gguf file plus manifest.json."],
        )

    def run(self, ctx: RunContext) -> StageResult:
        quantized = ctx.input_artifact(stage="quantize")
        if quantized is None:
            raise StageError("package requires a 'quantized_model' artifact; run quantize first.")
        tokenizer = ctx.input_artifact(stage="tokenizer-train")
        if tokenizer is None:
            raise StageError("package requires a 'tokenizer' artifact; run tokenizer-train first.")
        if ctx.run is None:
            raise StageError("package cannot run without an active run context.")

        manifest = _build_manifest(
            ctx, quantized_provenance=quantized.provenance, tokenizer=tokenizer
        )
        manifest_path = safe_resolve(ctx.workspace, "manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        package_path = safe_resolve(ctx.workspace, "package.zip")
        gguf_path = Path(quantized.path)
        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(gguf_path, arcname=gguf_path.name)
            archive.write(manifest_path, arcname="manifest.json")

        artifact = ctx.record_artifact(
            package_path,
            kind=ArtifactKind.PACKAGE,
            stage=self.name,
            provenance=manifest,
        )
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[artifact],
            metrics={"bytes": float(package_path.stat().st_size)},
            message=f"Packaged {gguf_path.name} + manifest into {package_path.name}.",
        )


def _build_manifest(
    ctx: RunContext, *, quantized_provenance: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    assert ctx.run is not None
    recommendation = ctx.project.recommendation
    return {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_run_id": ctx.run.record.id,
        "project_name": ctx.project.name,
        "model_size": ctx.project.model.name,
        "runtime": quantized_provenance.get("runtime", "gguf"),
        "quant_type": quantized_provenance.get("quant_type"),
        "vocab_size": quantized_provenance.get("vocab_size"),
        "layers": quantized_provenance.get("layers"),
        "tokenizer_sha256": tokenizer.sha256,
        "sizing_recommendation": recommendation.model_dump() if recommendation else None,
    }
