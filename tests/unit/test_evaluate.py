from core.editions import resolve_entitlements
from core.pipeline.context import RunContext
from core.pipeline.runs import RunStore
from core.pipeline.stages.data_ingest import DataIngestStage
from core.pipeline.stages.evaluate import EvaluateStage
from core.pipeline.stages.model_init import ModelInitStage
from core.pipeline.stages.pretrain import PretrainStage
from core.project import create_project

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog number {i}\n" for i in range(120))


def _build_checkpoint_and_val_split(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    entitlements = resolve_entitlements(env={})

    ingest_ctx = RunContext(
        project=project,
        entitlements=entitlements,
        run=run,
        params={"inputs": [str(source)], "val_ratio": 0.2},
    )
    DataIngestStage().run(ingest_ctx)

    tok_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"vocab_size": 300}
    )
    from core.pipeline.stages.tokenizer_train import TokenizerTrainStage

    TokenizerTrainStage().run(tok_ctx)

    init_ctx = RunContext(project=project, entitlements=entitlements, run=run, params={})
    ModelInitStage().run(init_ctx)

    train_ctx = RunContext(
        project=project,
        entitlements=entitlements,
        run=run,
        params={"max_steps": 20, "block_size": 8, "checkpoint_every": 10},
    )
    # PretrainStage always spawns a subprocess; checkpoints land under cwd/.sg2, so
    # give it an isolated cwd for this test.
    return project, run, entitlements, train_ctx


def test_evaluate_reports_perplexity_and_accuracy(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project, run, entitlements, train_ctx = _build_checkpoint_and_val_split(tmp_path)
    PretrainStage().run(train_ctx)

    eval_ctx = RunContext(project=project, entitlements=entitlements, run=run, params={})
    result = EvaluateStage().run(eval_ctx)

    assert result.status == "succeeded"
    assert result.metrics["perplexity"] > 1.0
    assert 0.0 <= result.metrics["top1_accuracy"] <= 1.0
    artifact = result.artifacts[0]
    assert artifact.kind.value == "eval_report"
    assert artifact.provenance["val_blocks"] > 0


def test_evaluate_requires_val_split(tmp_path, monkeypatch) -> None:
    import pytest

    from core.pipeline.base import StageError
    from core.pipeline.stages.data_ingest import DataIngestStage as _DI
    from core.pipeline.stages.tokenizer_train import TokenizerTrainStage

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    entitlements = resolve_entitlements(env={})

    ingest_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"inputs": [str(source)]}
    )
    _DI().run(ingest_ctx)  # no val_ratio -> no val split
    tok_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"vocab_size": 300}
    )
    TokenizerTrainStage().run(tok_ctx)
    init_ctx = RunContext(project=project, entitlements=entitlements, run=run, params={})
    ModelInitStage().run(init_ctx)
    train_ctx = RunContext(
        project=project,
        entitlements=entitlements,
        run=run,
        params={"max_steps": 5, "block_size": 8},
    )
    PretrainStage().run(train_ctx)

    eval_ctx = RunContext(project=project, entitlements=entitlements, run=run, params={})
    with pytest.raises(StageError, match="val-ratio"):
        EvaluateStage().run(eval_ctx)
