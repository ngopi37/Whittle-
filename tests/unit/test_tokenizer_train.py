import json

import pytest

from core.editions import resolve_entitlements
from core.pipeline.base import StageError
from core.pipeline.context import RunContext
from core.pipeline.runs import RunStore
from core.pipeline.stages.data_ingest import DataIngestStage
from core.pipeline.stages.tokenizer_train import TokenizerTrainStage
from core.project import create_project

_CORPUS = "".join(f"the quick brown fox jumps over the lazy dog {i}\n" for i in range(50))


def _ingest_then_tokenize(tmp_path, corpus: str, **params):
    source = tmp_path / "corpus.txt"
    source.write_text(corpus, encoding="utf-8")
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    entitlements = resolve_entitlements(env={})
    ingest_ctx = RunContext(
        project=project, entitlements=entitlements, run=run, params={"inputs": [str(source)]}
    )
    DataIngestStage().run(ingest_ctx)
    train_ctx = RunContext(project=project, entitlements=entitlements, run=run, params=params)
    return TokenizerTrainStage().run(train_ctx), run


def test_tokenizer_train_produces_bpe_artifact(tmp_path) -> None:
    result, run = _ingest_then_tokenize(tmp_path, _CORPUS, vocab_size=300)

    assert result.status == "succeeded"
    artifact = result.artifacts[0]
    assert artifact.kind.value == "tokenizer"
    assert artifact.provenance["algorithm"] == "bpe"
    tokenizer_path = run.artifacts_dir / "tokenizer.json"
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert payload["model"]["type"] == "BPE"
    assert result.metrics["vocab_size"] <= 300
    assert result.metrics["documents"] == 50


def test_tokenizer_train_defaults_vocab_size_to_model_config(tmp_path) -> None:
    result, _ = _ingest_then_tokenize(tmp_path, _CORPUS)

    # The 20M catalog entry declares vocabulary_size=16000; a tiny corpus can't fill
    # it, but the trainer must never exceed the requested ceiling.
    assert result.metrics["vocab_size"] <= 16000


def test_tokenizer_train_unigram_algorithm(tmp_path) -> None:
    result, run = _ingest_then_tokenize(tmp_path, _CORPUS, algorithm="unigram", vocab_size=64)

    assert result.status == "succeeded"
    tokenizer_path = run.artifacts_dir / "tokenizer.json"
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert payload["model"]["type"] == "Unigram"


def test_tokenizer_train_rejects_unknown_algorithm(tmp_path) -> None:
    with pytest.raises(StageError, match="Unknown tokenizer algorithm"):
        _ingest_then_tokenize(tmp_path, _CORPUS, algorithm="wordpiece")


def test_tokenizer_train_requires_prior_dataset(tmp_path) -> None:
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    ctx = RunContext(project=project, entitlements=resolve_entitlements(env={}), run=run, params={})
    with pytest.raises(StageError, match="run data-ingest first"):
        TokenizerTrainStage().run(ctx)


def test_tokenizer_train_rejects_empty_dataset(tmp_path) -> None:
    with pytest.raises(StageError, match="0 documents"):
        _ingest_then_tokenize(tmp_path, "\n\n")
