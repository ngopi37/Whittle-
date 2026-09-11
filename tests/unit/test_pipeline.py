import pytest

from core.editions import Edition, resolve_entitlements
from core.pipeline.base import StageError, StageNotImplemented
from core.pipeline.context import RunContext
from core.pipeline.pipeline import Pipeline
from core.pipeline.runs import RunStore
from core.pipeline.stages.deferred import DeferredStage
from core.project import create_project
from schemas.hardware import CpuProfile, GpuProfile, HardwareProfile, MemoryProfile, StorageProfile
from schemas.pipeline import STAGE_ORDER


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Windows",
        architecture="x86_64",
        cpu=CpuProfile(vendor="Intel", model="i7", physical_cores=6, logical_cores=12),
        memory=MemoryProfile(total_gb=16, available_gb=12),
        gpu=GpuProfile(),
        storage=StorageProfile(capacity_gb=500, free_gb=100),
    )


def _pipeline(tmp_path, edition="free") -> Pipeline:
    return Pipeline(
        run_store=RunStore(root=tmp_path / "runs"),
        entitlements=resolve_entitlements(env={"SG2_EDITION": edition}),
    )


def test_plan_covers_every_canonical_stage(tmp_path) -> None:
    project = create_project("demo", "20M", hardware=_hardware())
    plan = _pipeline(tmp_path).plan(project)
    assert tuple(p.stage for p in plan.stages) == STAGE_ORDER
    assert plan.total_estimated_memory_gb > 0
    assert sum(1 for p in plan.stages if p.implemented) == 8


def test_run_data_ingest_then_stop(tmp_path) -> None:
    source = tmp_path / "d.txt"
    source.write_text("hello\nworld\n", encoding="utf-8")
    project = create_project("demo", "20M")
    handle = _pipeline(tmp_path).run(
        project, stages=["data-ingest"], params={"inputs": [str(source)]}
    )
    assert handle.record.status == "succeeded"
    assert handle.record.stages["data-ingest"] == "succeeded"
    assert handle.record.artifacts[0].kind.value == "dataset"
    assert handle.events_path.read_text(encoding="utf-8").count("\n") >= 3


def test_deferred_stage_mechanism_still_works(tmp_path) -> None:
    # No stage in the default registry is deferred anymore (P2-P5 all landed), but
    # DeferredStage/StageNotImplemented remain live infrastructure for a future
    # phase's stages — exercise them directly rather than through the registry.
    project = create_project("demo", "20M")
    run = RunStore(root=tmp_path / "runs").create_run(project_name="demo", model_size="20M")
    ctx = RunContext(project=project, entitlements=resolve_entitlements(env={}), run=run, params={})
    stage = DeferredStage(name="pretrain", roadmap_phase="P9")
    assert stage.plan(ctx).implemented is False
    with pytest.raises(StageNotImplemented, match="roadmap P9"):
        stage.run(ctx)


def test_evaluate_without_pretrain_reports_a_clean_error(tmp_path) -> None:
    project = create_project("demo", "20M")
    with pytest.raises(StageError, match="run pretrain first"):
        _pipeline(tmp_path).run(project, stages=["evaluate"])


def test_unknown_stage_is_rejected(tmp_path) -> None:
    project = create_project("demo", "20M")
    with pytest.raises(KeyError, match="Unknown stage"):
        _pipeline(tmp_path).run(project, stages=["nonsense"])  # type: ignore[list-item]


def test_remote_run_requires_paid_edition(tmp_path) -> None:
    project = create_project("demo", "20M")
    with pytest.raises(Exception, match="remote-training"):
        _pipeline(tmp_path, edition="free").run(project, stages=["pretrain"], remote=True)


def test_run_store_lists_runs(tmp_path) -> None:
    store = RunStore(root=tmp_path / "runs")
    store.create_run(project_name="a", model_size="20M")
    store.create_run(project_name="b", model_size="50M")
    assert {r.project_name for r in store.list_runs()} == {"a", "b"}


def test_entitlements_pro_unlocks_large_models() -> None:
    assert resolve_entitlements(env={"SG2_EDITION": "pro"}).edition is Edition.PRO
