"""Tests for the pretrain worker's training loop, run in-process (no subprocess).

``core.pipeline.stages.pretrain.PretrainStage`` always spawns a real subprocess (see
``test_pretrain.py`` for that end-to-end path); calling ``train_worker.run`` directly
here keeps these tests fast while still exercising real torch training and real
checkpoint resume.
"""

from __future__ import annotations

from core.editions import resolve_entitlements
from core.pipeline.context import RunContext
from core.pipeline.runs import RunStore
from core.pipeline.stages.data_ingest import DataIngestStage
from core.pipeline.stages.model_init import ModelInitStage
from core.pipeline.stages.tokenizer_train import TokenizerTrainStage
from core.pipeline.workers import train_worker
from core.project import create_project
from schemas.pipeline import JobSpec

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog number {i}\n" for i in range(80))


def _build_dataset_tokenizer_and_model(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    entitlements = resolve_entitlements(env={})

    ingest_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"inputs": [str(source)]}
    )
    dataset = DataIngestStage().run(ingest_ctx).artifacts[0]

    tok_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"vocab_size": 300}
    )
    tokenizer = TokenizerTrainStage().run(tok_ctx).artifacts[0]

    init_ctx = RunContext(project=project, entitlements=entitlements, run=run, params={})
    model_init = ModelInitStage().run(init_ctx).artifacts[0]

    return dataset, tokenizer, model_init


def _job(*, dataset, tokenizer, model_init, checkpoint_dir, **overrides) -> JobSpec:
    params = {
        "dataset_path": dataset.path,
        "model_init_path": model_init.path,
        "tokenizer_path": tokenizer.path,
        "checkpoint_dir": str(checkpoint_dir),
        "block_size": 8,
        "batch_size": 2,
        "learning_rate": 1e-2,
        "gradient_accumulation_steps": 1,
        "mixed_precision": False,
        "max_steps": 20,
        "checkpoint_every": 5,
        "seed": 7,
    }
    params.update(overrides)
    return JobSpec(
        schema_version=1, job_type="stage", stage="pretrain", project_name="demo", params=params
    )


def test_worker_trains_and_loss_trends_down(tmp_path) -> None:
    dataset, tokenizer, model_init = _build_dataset_tokenizer_and_model(tmp_path)
    job = _job(
        dataset=dataset,
        tokenizer=tokenizer,
        model_init=model_init,
        checkpoint_dir=tmp_path / "ckpt",
        max_steps=60,
    )
    events = []
    train_worker.run(job, emit=lambda kind, payload: events.append((kind, payload)))

    losses = [payload["loss"] for kind, payload in events if kind == "metric"]
    assert len(losses) == 60
    first_avg = sum(losses[:5]) / 5
    last_avg = sum(losses[-5:]) / 5
    assert last_avg < first_avg

    done = next(payload for kind, payload in events if kind == "done")
    assert done["steps"] == 60
    assert (tmp_path / "ckpt" / "checkpoint_last.pt").is_file()


def test_worker_resumes_from_checkpoint_instead_of_restarting(tmp_path) -> None:
    dataset, tokenizer, model_init = _build_dataset_tokenizer_and_model(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"

    first_job = _job(
        dataset=dataset, tokenizer=tokenizer, model_init=model_init,
        checkpoint_dir=checkpoint_dir, max_steps=10,
    )
    train_worker.run(first_job, emit=lambda *_: None)

    events = []
    second_job = _job(
        dataset=dataset, tokenizer=tokenizer, model_init=model_init,
        checkpoint_dir=checkpoint_dir, max_steps=20,
    )
    train_worker.run(second_job, emit=lambda kind, payload: events.append((kind, payload)))

    resume_logs = [
        p["message"] for k, p in events if k == "log" and "Resuming" in p.get("message", "")
    ]
    assert resume_logs and "step 10" in resume_logs[0]
    steps_trained_second_run = len([1 for k, p in events if k == "metric"])
    assert steps_trained_second_run == 10  # resumed at 10, ran to max_steps 20
    done = next(p for k, p in events if k == "done")
    assert done["steps"] == 20
