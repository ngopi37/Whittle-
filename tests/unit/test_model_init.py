import pytest

from core.editions import resolve_entitlements
from core.pipeline.base import StageError
from core.pipeline.context import RunContext
from core.pipeline.runs import RunStore
from core.pipeline.stages.data_ingest import DataIngestStage
from core.pipeline.stages.model_init import ModelInitStage
from core.pipeline.stages.tokenizer_train import TokenizerTrainStage
from core.project import create_project

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog {i}\n" for i in range(50))


def _tokenizer_ctx(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(_CORPUS, encoding="utf-8")
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    entitlements = resolve_entitlements(env={})
    ingest_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"inputs": [str(source)]}
    )
    DataIngestStage().run(ingest_ctx)
    tok_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"vocab_size": 300}
    )
    TokenizerTrainStage().run(tok_ctx)
    return RunContext(project=project, entitlements=entitlements, run=run, params={})


def test_model_init_builds_model_matching_tokenizer_vocab(tmp_path) -> None:
    ctx = _tokenizer_ctx(tmp_path)

    result = ModelInitStage().run(ctx)

    assert result.status == "succeeded"
    artifact = result.artifacts[0]
    assert artifact.kind.value == "model_init"
    tokenizer_artifact = ctx.input_artifact(stage="tokenizer-train")
    assert artifact.provenance["vocab_size"] == tokenizer_artifact.provenance["vocab_size_actual"]
    assert result.metrics["parameters"] > 0


def test_model_init_requires_tokenizer(tmp_path) -> None:
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    ctx = RunContext(project=project, entitlements=resolve_entitlements(env={}), run=run, params={})
    with pytest.raises(StageError, match="run tokenizer-train first"):
        ModelInitStage().run(ctx)
