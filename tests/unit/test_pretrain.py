"""End-to-end ``pretrain`` test: the real worker subprocess, not a direct call.

Slower than ``test_train_worker.py`` (pays subprocess + torch-import startup once),
but it is the only test that actually exercises the out-of-process wiring
``PretrainStage`` is required to use.
"""

from __future__ import annotations

from core.editions import resolve_entitlements
from core.pipeline.pipeline import Pipeline
from core.pipeline.runs import RunStore
from core.project import create_project

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog number {i}\n" for i in range(80))


def test_pretrain_runs_end_to_end_via_worker_subprocess(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # checkpoints live under Path.cwd()/.sg2/checkpoints/<project>
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")

    pipeline = Pipeline(
        run_store=RunStore(root=tmp_path / "runs"),
        entitlements=resolve_entitlements(env={"SG2_EDITION": "free"}),
    )
    handle = pipeline.run(
        project,
        stages=["data-ingest", "tokenizer-train", "model-init", "pretrain"],
        params={
            "inputs": [str(source)],
            "vocab_size": 300,
            "max_steps": 6,
            "block_size": 8,
            "checkpoint_every": 3,
        },
    )

    assert handle.record.status == "succeeded"
    checkpoint_artifact = handle.record.artifacts[-1]
    assert checkpoint_artifact.kind.value == "checkpoint"
    assert checkpoint_artifact.provenance["steps"] == 6
    checkpoint_path = tmp_path / ".sg2" / "checkpoints" / "demo" / "pretrain" / "checkpoint_last.pt"
    assert checkpoint_path.is_file()
