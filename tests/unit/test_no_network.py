"""Proves "no network call, ever" rather than just asserting it in docs and code.

Runs the real pipeline stage logic — data-ingest through the on-device smoke test
— with the actual socket-level ``connect``/``getaddrinfo`` primitives patched to
raise the instant anything tries to reach the network (``tests.network_guard``).
If any dependency (torch, tokenizers, huggingface_hub, gguf, llama-cpp-python, or
anything pulled in later) ever starts phoning home, this test fails loudly instead
of the claim silently going stale.

Scope: this exercises stage *logic*, called in-process. ``pretrain``'s real path
runs in a worker subprocess (``PretrainStage``, exercised for real — subprocess
and all — in ``test_pretrain.py``); spawning that subprocess via
``subprocess.Popen`` doesn't touch the network by construction, so what actually
matters for the "no network" claim is whether the training *code* makes network
calls, which this test checks by calling ``train_worker.run`` directly instead of
through the subprocess boundary.
"""

from __future__ import annotations

import socket
import zipfile
from pathlib import Path
from typing import Any

import pytest

from core.editions import resolve_entitlements
from core.pipeline.context import RunContext
from core.pipeline.runs import RunStore
from core.pipeline.stages._dataset import select_dataset_artifact
from core.pipeline.stages.data_ingest import DataIngestStage
from core.pipeline.stages.evaluate import EvaluateStage
from core.pipeline.stages.model_init import ModelInitStage
from core.pipeline.stages.package import PackageStage
from core.pipeline.stages.quantize import QuantizeStage
from core.pipeline.stages.tokenizer_train import TokenizerTrainStage
from core.pipeline.workers import train_worker
from core.project import create_project
from schemas.pipeline import ArtifactKind, JobSpec
from tests.network_guard import NetworkBlockedError, block_network

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog number {i}\n" for i in range(150))


def test_guard_actually_blocks_a_real_connection_attempt() -> None:
    """Sanity-check the guard itself: prove it isn't a silent no-op."""
    with block_network(), pytest.raises(NetworkBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("example.com", 80))


def test_full_pipeline_makes_no_network_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    entitlements = resolve_entitlements(env={})

    def ctx(**params: object) -> RunContext:
        return RunContext(project=project, entitlements=entitlements, run=run, params=params)

    with block_network():
        DataIngestStage().run(ctx(inputs=[str(source)], val_ratio=0.2))
        TokenizerTrainStage().run(ctx(vocab_size=300))
        ModelInitStage().run(ctx())

        dataset = select_dataset_artifact(ctx(), role="train")
        tokenizer = ctx().input_artifact(stage="tokenizer-train")
        model_init = ctx().input_artifact(stage="model-init")
        assert dataset is not None and tokenizer is not None and model_init is not None

        checkpoint_dir = tmp_path / "checkpoints"
        job = JobSpec(
            schema_version=1,
            job_type="stage",
            stage="pretrain",
            project_name="demo",
            params={
                "dataset_path": dataset.path,
                "model_init_path": model_init.path,
                "tokenizer_path": tokenizer.path,
                "checkpoint_dir": str(checkpoint_dir),
                "block_size": 8,
                "batch_size": 2,
                "learning_rate": 1e-2,
                "max_steps": 6,
                "checkpoint_every": 3,
                "seed": 7,
            },
        )
        events: list[tuple[str, dict[str, object]]] = []
        train_worker.run(job, emit=lambda kind, payload: events.append((kind, payload)))
        done = next(payload for kind, payload in events if kind == "done")
        checkpoint_artifact = run.record_artifact(
            Path(str(done["checkpoint_path"])),
            kind=ArtifactKind.CHECKPOINT,
            stage="pretrain",
            provenance={"final_loss": done["final_loss"], "steps": done["steps"]},
        )
        assert checkpoint_artifact.path

        EvaluateStage().run(ctx())
        QuantizeStage().run(ctx(quant_type="q8_0"))
        package_result = PackageStage().run(ctx())

        package_path = package_result.artifacts[0].path
        extract_dir = tmp_path / "smoke_extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(package_path) as archive:
            names = archive.namelist()
            archive.extractall(extract_dir)
        assert any(name.endswith(".gguf") for name in names)

        from llama_cpp import Llama

        gguf_path = next(extract_dir.glob("*.gguf"))
        llm = Llama(model_path=str(gguf_path), n_ctx=64, verbose=False)
        completion: Any = llm("the quick brown", max_tokens=8, temperature=0.0, stream=False)
        assert completion["choices"][0]["text"]
