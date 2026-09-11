import json

import pytest

from core.editions import resolve_entitlements
from core.pipeline.base import StageError
from core.pipeline.context import RunContext
from core.pipeline.runs import RunStore
from core.pipeline.stages.data_ingest import DataIngestStage
from core.project import create_project


def _context(tmp_path, inputs, **params):
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    return RunContext(
        project=project,
        entitlements=resolve_entitlements(env={}),
        run=run,
        params={"inputs": inputs, **params},
    )


def test_ingest_normalizes_jsonl_and_records_provenance(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
    ctx = _context(tmp_path, [str(source)])

    result = DataIngestStage().run(ctx)

    assert result.status == "succeeded"
    assert result.metrics["documents"] == 2
    artifact = result.artifacts[0]
    lines = (tmp_path / "runs").glob("*/artifacts/dataset.jsonl")
    dataset = next(lines).read_text(encoding="utf-8").splitlines()
    assert json.loads(dataset[0]) == {"text": "one"}
    assert artifact.provenance["sources"][0]["documents"] == 2
    assert artifact.sha256


def test_ingest_wraps_plain_text_lines(tmp_path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text("first line\n\nsecond line\n", encoding="utf-8")
    ctx = _context(tmp_path, [str(source)])

    result = DataIngestStage().run(ctx)

    assert result.metrics["documents"] == 2


def test_ingest_rejects_invalid_json(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text("{not json}\n", encoding="utf-8")
    ctx = _context(tmp_path, [str(source)])
    with pytest.raises(StageError, match="not valid JSON"):
        DataIngestStage().run(ctx)


def test_ingest_requires_a_source(tmp_path) -> None:
    ctx = _context(tmp_path, [])
    with pytest.raises(StageError, match="at least one"):
        DataIngestStage().run(ctx)


def test_ingest_rejects_unknown_suffix(tmp_path) -> None:
    source = tmp_path / "data.bin"
    source.write_text("x", encoding="utf-8")
    ctx = _context(tmp_path, [str(source)])
    with pytest.raises(StageError, match="Unsupported input type"):
        DataIngestStage().run(ctx)


def test_ingest_dedups_exact_duplicates_by_default(tmp_path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("one\ntwo\none\n", encoding="utf-8")
    ctx = _context(tmp_path, [str(source)])

    result = DataIngestStage().run(ctx)

    assert result.metrics["documents"] == 2
    assert result.metrics["duplicates_removed"] == 1
    assert result.artifacts[0].provenance["role"] == "all"


def test_ingest_dedup_can_be_disabled(tmp_path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("one\ntwo\none\n", encoding="utf-8")
    ctx = _context(tmp_path, [str(source)], dedup=False)

    result = DataIngestStage().run(ctx)

    assert result.metrics["documents"] == 3
    assert result.metrics["duplicates_removed"] == 0


def test_ingest_val_split_is_deterministic_and_disjoint(tmp_path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("".join(f"line {i}\n" for i in range(200)), encoding="utf-8")

    result_a = DataIngestStage().run(_context(tmp_path / "a", [str(source)], val_ratio=0.3))
    result_b = DataIngestStage().run(_context(tmp_path / "b", [str(source)], val_ratio=0.3))

    assert len(result_a.artifacts) == 2
    train_role = {a.provenance["role"] for a in result_a.artifacts}
    assert train_role == {"train", "val"}
    assert result_a.metrics["documents"] + result_a.metrics["val_documents"] == 200
    assert result_a.metrics["val_documents"] > 0
    # Same input + same seed -> same split, regardless of run.
    assert result_a.metrics["val_documents"] == result_b.metrics["val_documents"]


def test_ingest_rejects_invalid_val_ratio(tmp_path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("one\n", encoding="utf-8")
    ctx = _context(tmp_path, [str(source)], val_ratio=1.0)
    with pytest.raises(StageError, match="val_ratio must be in"):
        DataIngestStage().run(ctx)


def test_ingest_reads_parquet_text_column(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "data.parquet"
    table = pa.table({"text": ["alpha", "beta", ""]})
    pq.write_table(table, source)
    ctx = _context(tmp_path, [str(source)])

    result = DataIngestStage().run(ctx)

    assert result.metrics["documents"] == 2
    assert result.artifacts[0].provenance["sources"][0]["format"] == "parquet"


def test_ingest_parquet_without_text_column_is_rejected(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "data.parquet"
    pq.write_table(pa.table({"body": ["alpha"]}), source)
    ctx = _context(tmp_path, [str(source)])
    with pytest.raises(StageError, match="no 'text' column"):
        DataIngestStage().run(ctx)
