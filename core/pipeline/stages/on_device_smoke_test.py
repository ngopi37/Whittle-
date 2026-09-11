"""Load the packaged GGUF file with real ``llama.cpp`` and generate a few tokens.

This is the actual verification that ``quantize``/``package`` produced a file a
real on-device runtime can use — not just that this codebase's own writer ran
without raising. ``llama_cpp`` is imported lazily inside ``run()``.
"""

from __future__ import annotations

import time
import zipfile
from typing import Any

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.pipeline.stages._params import int_param
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult

_DEFAULT_N_CTX = 128
_DEFAULT_MAX_TOKENS = 32
_DEFAULT_PROMPT = "Hello"


class OnDeviceSmokeTestStage(Stage):
    """Load the packaged model with llama.cpp and generate a few tokens."""

    name = "on-device-smoke-test"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.PACKAGE],
            outputs=[],
            notes=["Real llama.cpp load + generation via llama-cpp-python, not a format check."],
        )

    def run(self, ctx: RunContext) -> StageResult:
        package = ctx.input_artifact(stage="package")
        if package is None:
            raise StageError(
                "on-device-smoke-test requires a 'package' artifact; run package first."
            )

        extract_dir = safe_resolve(ctx.workspace, "smoke_test_extract")
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(package.path) as archive:
            archive.extractall(extract_dir)
        gguf_files = list(extract_dir.glob("*.gguf"))
        if not gguf_files:
            raise StageError(f"package archive {package.path} contains no .gguf file.")
        gguf_path = gguf_files[0]

        prompt = str(ctx.params.get("prompt") or _DEFAULT_PROMPT)
        max_tokens = int_param(ctx.params, "max_tokens", _DEFAULT_MAX_TOKENS)
        n_ctx = int_param(ctx.params, "n_ctx", _DEFAULT_N_CTX)

        try:
            from llama_cpp import Llama

            llm = Llama(model_path=str(gguf_path), n_ctx=n_ctx, verbose=False)
            start = time.perf_counter()
            completion: Any = llm(prompt, max_tokens=max_tokens, temperature=0.0, stream=False)
            elapsed = max(time.perf_counter() - start, 1e-6)
        except Exception as exc:
            raise StageError(
                f"on-device-smoke-test failed to load/run the package: {exc}"
            ) from exc

        text = completion["choices"][0]["text"]
        generated_tokens = int(completion.get("usage", {}).get("completion_tokens", max_tokens))
        tokens_per_second = generated_tokens / elapsed

        rss_mb = _process_rss_mb()
        ctx.log(f"Generated: {prompt!r} -> {text!r}", stage=self.name)

        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[],
            metrics={
                "tokens_per_second": tokens_per_second,
                "generated_tokens": float(generated_tokens),
                "rss_mb": rss_mb,
            },
            message=(
                f"{generated_tokens} token(s) in {elapsed:.2f}s "
                f"({tokens_per_second:.1f} tok/s)."
            ),
        )


def _process_rss_mb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss) / 2**20
