"""End-to-end P4/P5 (GGUF-only) verification: the real pipeline, start to finish.

One test trains a tiny real checkpoint, quantizes it to GGUF, packages it, and then
loads the package with real ``llama.cpp`` (via ``llama-cpp-python``) and generates
tokens — the actual proof that ``quantize``/``package`` produce a file a real
on-device runtime accepts, not just that our own writer ran without raising.
"""

from __future__ import annotations

import pytest

from core.editions import resolve_entitlements
from core.pipeline.base import StageError
from core.pipeline.context import RunContext
from core.pipeline.pipeline import Pipeline
from core.pipeline.runs import RunStore
from core.pipeline.stages.on_device_smoke_test import OnDeviceSmokeTestStage
from core.pipeline.stages.package import PackageStage
from core.pipeline.stages.quantize import QuantizeStage
from core.project import create_project

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog number {i}\n" for i in range(150))


def test_full_pipeline_end_to_end_produces_a_working_gguf_package(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # pretrain checkpoints live under cwd/.sg2
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")

    pipeline = Pipeline(
        run_store=RunStore(root=tmp_path / "runs"),
        entitlements=resolve_entitlements(env={"SG2_EDITION": "free"}),
    )
    handle = pipeline.run(
        project,
        stages=[
            "data-ingest",
            "tokenizer-train",
            "model-init",
            "pretrain",
            "evaluate",
            "quantize",
            "package",
            "on-device-smoke-test",
        ],
        params={
            "inputs": [str(source)],
            "vocab_size": 300,
            "val_ratio": 0.2,
            "max_steps": 20,
            "block_size": 8,
            "checkpoint_every": 10,
            "quant_type": "q8_0",
            "prompt": "the quick brown",
            "max_tokens": 8,
        },
    )

    assert handle.record.status == "succeeded"
    kinds = [a.kind.value for a in handle.record.artifacts]
    assert "checkpoint" in kinds
    assert "eval_report" in kinds
    assert "quantized_model" in kinds
    assert "package" in kinds

    smoke_status = handle.record.stages["on-device-smoke-test"]
    assert smoke_status == "succeeded"


def test_quantize_requires_checkpoint(tmp_path) -> None:
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    ctx = RunContext(project=project, entitlements=resolve_entitlements(env={}), run=run, params={})
    with pytest.raises(StageError, match="run pretrain first"):
        QuantizeStage().run(ctx)


def test_package_requires_quantized_model(tmp_path) -> None:
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    ctx = RunContext(project=project, entitlements=resolve_entitlements(env={}), run=run, params={})
    with pytest.raises(StageError, match="run quantize first"):
        PackageStage().run(ctx)


def test_on_device_smoke_test_requires_package(tmp_path) -> None:
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    ctx = RunContext(project=project, entitlements=resolve_entitlements(env={}), run=run, params={})
    with pytest.raises(StageError, match="run package first"):
        OnDeviceSmokeTestStage().run(ctx)
