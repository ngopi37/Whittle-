# Architecture

A thin application layer over independent core services, with all cross-module
contracts as Pydantic models in `schemas/`.

## Contracts — `schemas/`

- `schemas/hardware.py`: normalized hardware profile.
- `schemas/model.py`: `ModelConfig` (architecture) and `TrainingConfig` (knobs).
- `schemas/pipeline.py`: stage, run, artifact, and `JobSpec` v1 contracts.

## Core services — `core/`

- `core.hardware`: measured local hardware normalization (Windows CIM + registry +
  `IsProcessorFeaturePresent`, degrading elsewhere).
- `core.catalog`: the single source of truth for supported model sizes (20M–500M),
  their YAML profiles, and the edition each requires.
- `core.sizing`: conservative, explainable resource breakdowns as a function of the
  hardware profile *and* the project's model/training config; status, confidence,
  disk gating, duration ranges, and reasons.
- `core.models`: YAML-backed initial architecture profile generation.
- `core.project`: validated local project persistence with optional embedded
  hardware/recommendation evidence; `refresh_recommendation`.
- `core.editions`: offline edition/entitlement resolution and feature gating.
- `core.safety`: path-traversal-safe resolution; archive extraction contract.
- `core.pipeline`: the ordered build — `Stage` interface, `RunStore`, `Pipeline`
  orchestrator, and the stage registry. See [pipeline.md](pipeline.md).
- `core.training`: the model architecture (`model.py`, a Llama-style decoder) and
  dataset tokenization (`data.py`), shared by `model_init`/`pretrain`/`evaluate`.
  Not imported at module load by any stage — only lazily inside `run()` — so no
  CLI/desktop invocation pays `torch`'s import cost unless that stage actually runs.
- `core.runtime`: on-device export. `gguf_export.py` writes a checkpoint straight
  to GGUF (fp16/Q8_0/Q4_0) using the `gguf` package's own reference quantization —
  currently the only implemented runtime target; see
  [runtime-targets.md](runtime-targets.md).

## Applications — `apps/`

- `apps.cli`, `apps.desktop`: presentation only. Hardware detection, sizing, catalog
  loading, project persistence, edition resolution, and pipeline orchestration all
  live in `core`. The desktop shell stays Tkinter (stdlib, single PyInstaller build).

## Dependency direction

`apps → core → schemas`. `core.sizing` and `core.pipeline` depend on `schemas`, not
on each other's UI. `core.pipeline.workers.train_worker` depends on
`schemas/pipeline.py` (`JobSpec`) and runs out-of-process — `PretrainStage` never
imports `torch` into the shell; `model-init`/`evaluate`/`quantize` do import it,
but lazily inside `run()`, never at module load.
