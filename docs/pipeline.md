# Pipeline Architecture

The build is a fixed, ordered sequence of stages. Each stage declares a plan
(cheap, side-effect free) and, when run, writes artifacts into a run directory
through a `RunContext`. The orchestrator is `core.pipeline.Pipeline`; contracts are
in `schemas/pipeline.py`.

## Canonical stages

```
data-ingest → tokenizer-train → model-init → pretrain
            → evaluate → quantize → package → on-device-smoke-test
```

`STAGE_ORDER` in `schemas/pipeline.py` is the single source of truth for the order;
`core/pipeline/registry.py` binds each name to an implementation. Every canonical
stage is implemented (`quantize`/`package`/`on-device-smoke-test` support the
GGUF/llama.cpp runtime target only for now — see
[runtime-targets.md](runtime-targets.md)). `DeferredStage` / `StageNotImplemented`
remain live infrastructure for a future phase's stages; see
`tests/unit/test_pipeline.py::test_deferred_stage_mechanism_still_works` for how
to exercise them directly.

## Stage interface

```python
class Stage(abc.ABC):
    name: StageName
    def plan(self, ctx: RunContext) -> StagePlan: ...   # no side effects, run may be None
    def run(self, ctx: RunContext) -> StageResult: ...   # writes artifacts via ctx
```

`StagePlan` carries declared `inputs`/`outputs` (as `ArtifactKind`), rough
`estimated_memory_gb` / `estimated_disk_gb`, a `required_feature` for edition gating,
and `implemented`.

## Runs and the run store

`RunStore` (default root `./.sg2/runs`) lays each run out as:

```
<root>/<run_id>/
    run.json        # RunRecord: status, per-stage status, artifact list
    events.jsonl    # append-only RunEvent log
    artifacts/      # stage outputs (dataset.jsonl, tokenizer/, checkpoints/, …)
```

`run_id` is `YYYYMMDD-HHMMSS-<hex>`. Artifacts are recorded with a sha256 and byte
size so a later stage (or a registry) can verify them.

## Artifacts

`Artifact` = `{id, kind, path, sha256, bytes, produced_by_stage, provenance}`.
`provenance` is free-form JSON; `data-ingest` records each source file's path, size,
mtime, line count, and format there.

## The job protocol (workers)

`pretrain` — the only genuinely long-running stage — runs **outside** the UI/CLI
process, as a worker (`core.pipeline.workers.train_worker`). The boundary is
`JobSpec` (versioned: `schema_version = 1`):

```python
JobSpec(schema_version=1, job_type="stage", stage="pretrain",
        project_name="…", params={…}, resources={"memory_gb": …})
```

A worker consumes a `JobSpec`, streams `JobEvent`s (`progress` / `metric` / `log` /
`artifact` / `done` / `error`) back over stdout or a file, and never shares process
state with the caller. Bump `schema_version` on any breaking change.

`quantize` deliberately does **not** use a worker: for the free tier's model
sizes, GGUF export is a handful of numpy-vectorized tensor writes (no training
loop), fast enough to run in-process with a lazy `torch`/`gguf` import — the
original plan to also make it a worker was more caution than the actual workload
needs. Revisit if a future runtime target's export turns out to be slow.

## Edition gating

`Pipeline._check_gating` calls `Entitlements.require(...)` for any stage with a
`required_feature`, and — when `remote=True` — for the scale-out feature mapped in
`REMOTE_STAGE_FEATURES`. Local runs of every stage are free; only the remote/registry/
deploy variants are gated. See [product/editions.md](product/editions.md).

## Adding a stage

1. Implement `Stage` in `core/pipeline/stages/`.
2. Replace the matching `DeferredStage` in `core/pipeline/registry.py`.
3. Keep `plan()` free of side effects and tolerant of `ctx.run is None`.
4. Write artifacts under `ctx.workspace`; register them with `ctx.record_artifact`.
5. Add unit tests mirroring `tests/unit/test_data_ingest.py`.
