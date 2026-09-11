"""Sequential orchestration of the local model build."""

from __future__ import annotations

from core.editions import Entitlements, resolve_entitlements
from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.pipeline.registry import REMOTE_STAGE_FEATURES, build_default_pipeline
from core.pipeline.runs import RunHandle, RunStore
from core.project import ProjectConfig
from core.sizing.engine import ModelFitEngine
from schemas.pipeline import PipelinePlan, StageName, StagePlan, StageResult


class Pipeline:
    """Plan and run the ordered build stages for a project."""

    def __init__(
        self,
        stages: list[Stage] | None = None,
        *,
        run_store: RunStore | None = None,
        entitlements: Entitlements | None = None,
    ) -> None:
        self.stages = stages if stages is not None else build_default_pipeline()
        self.run_store = run_store or RunStore()
        self.entitlements = entitlements or resolve_entitlements()

    def plan(
        self, project: ProjectConfig, *, params: dict[str, object] | None = None
    ) -> PipelinePlan:
        """Aggregate every stage's plan plus a whole-run resource estimate."""
        ctx = RunContext(
            project=project, entitlements=self.entitlements, run=None, params=params or {}
        )
        stage_plans: list[StagePlan] = [stage.plan(ctx) for stage in self.stages]

        training_memory = 0.0
        training_disk = 0.0
        if project.hardware is not None:
            fit = ModelFitEngine().evaluate(
                project.hardware,
                project.model.name,
                training=project.training,
                model_config=project.model,
            )
            training_memory = fit.memory_required_gb
            training_disk = fit.disk_required_gb

        total_memory = max([training_memory, *(p.estimated_memory_gb for p in stage_plans)])
        total_disk = training_disk + sum(p.estimated_disk_gb for p in stage_plans)
        return PipelinePlan(
            project_name=project.name,
            model_size=project.model.name,
            stages=stage_plans,
            total_estimated_memory_gb=round(total_memory, 2),
            total_estimated_disk_gb=round(total_disk, 2),
        )

    def run(
        self,
        project: ProjectConfig,
        *,
        stages: list[StageName] | None = None,
        params: dict[str, object] | None = None,
        remote: bool = False,
    ) -> RunHandle:
        """Run the requested stages in canonical order, stopping on the first failure."""
        selected = self._selected_stages(stages)
        self._check_gating(selected, remote=remote)

        handle = self.run_store.create_run(
            project_name=project.name, model_size=project.model.name
        )
        handle.set_status("running")
        ctx = RunContext(
            project=project, entitlements=self.entitlements, run=handle, params=params or {}
        )
        try:
            for stage in selected:
                handle.set_stage_status(stage.name, "running")
                handle.append_event(f"Starting stage '{stage.name}'.", stage=stage.name)
                result: StageResult = stage.run(ctx)
                handle.set_stage_status(stage.name, result.status)
                handle.append_event(
                    f"Stage '{stage.name}' {result.status}: {result.message}", stage=stage.name
                )
                if result.status != "succeeded":
                    handle.set_status("failed")
                    return handle
        except StageError as exc:
            handle.append_event(str(exc), level="error")
            handle.set_status("failed")
            raise
        handle.set_status("succeeded")
        return handle

    def _selected_stages(self, stages: list[StageName] | None) -> list[Stage]:
        if stages is None:
            return list(self.stages)
        wanted = set(stages)
        known = {stage.name for stage in self.stages}
        unknown = wanted - known
        if unknown:
            raise KeyError(f"Unknown stage(s): {', '.join(sorted(unknown))}")
        return [stage for stage in self.stages if stage.name in wanted]

    def _check_gating(self, selected: list[Stage], *, remote: bool) -> None:
        for stage in selected:
            required = getattr(stage, "required_feature", None)
            if required:
                self.entitlements.require(required)
            if remote and stage.name in REMOTE_STAGE_FEATURES:
                self.entitlements.require(REMOTE_STAGE_FEATURES[stage.name])
