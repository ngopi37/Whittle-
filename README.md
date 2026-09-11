# SG2 On-Device Model Builder

SG2 is a Windows-first, offline-capable environment that automates building a small
model to run **on the device in front of you**: profile the machine, prepare data,
train, evaluate, quantize, package for an on-device runtime, and smoke-test it —
locally, with no network dependency.

**Local is free, forever.** Paid editions add scale (remote/multi-GPU training), a
shared model registry, fleet deployment, teams, and governance. See
[docs/product/editions.md](docs/product/editions.md).

## Status

The full local pipeline (P1–P5) is shipped end to end — raw text in, a working
on-device package out — for the **GGUF/llama.cpp** runtime target. ExecuTorch and
TFLite/Core ML are deliberately deferred (different, heavy toolchains; see
[docs/runtime-targets.md](docs/runtime-targets.md)), not silently missing. See
[docs/product/roadmap.md](docs/product/roadmap.md).

## Quick start

Requires Python 3.12+. `torch` and `llama-cpp-python` are heavy (hundreds of MB);
use the CPU-only torch wheel unless you specifically want CUDA:

```powershell
python -m pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu

python -m apps.cli.main hardware                     # measured device profile
python -m apps.cli.main edition                      # active edition + features
python -m apps.cli.main recommend --batch-size 8 --mixed-precision

python -m apps.cli.main project create support-model --model 20M --with-hardware --output project.json
python -m apps.cli.main project show project.json
python -m apps.cli.main project refresh project.json # re-profile + recompute sizing

python -m apps.cli.main pipeline plan project.json   # the whole build, per-stage

# The whole build, one command: raw text -> a working GGUF package.
python -m apps.cli.main pipeline run project.json `
  --stage data-ingest --input corpus.jsonl --val-ratio 0.1 `
  --stage tokenizer-train --vocab-size 4000 `
  --stage model-init `
  --stage pretrain --max-steps 200 `
  --stage evaluate `
  --stage quantize --quant-type q8_0 `
  --stage package `
  --stage on-device-smoke-test --prompt "Hello" --max-tokens 32

pytest ; ruff check . ; mypy .
```

The desktop shell is `python -m apps.desktop.main` (Tkinter, no network). It opens
immediately and detects hardware on a worker thread.

## Architecture

The CLI and desktop are thin presentation over `core`: the profiler, sizing engine,
model catalog, project schema, edition resolver, and the pipeline orchestrator.
Cross-module contracts are Pydantic models in `schemas/`. See
[docs/architecture.md](docs/architecture.md), [docs/pipeline.md](docs/pipeline.md),
and [docs/hardware-sizing.md](docs/hardware-sizing.md).

## Editions

Resolved offline from `SG2_EDITION` or `~/.sg2/license.json`, default `free`. Model
sizes above 100M and any remote/registry/deploy capability require a paid edition;
the full local pipeline does not.

## Windows packaging

Deferred until the pipeline stages land. Target: a PyInstaller one-folder build,
documented in [docs/windows-build.md](docs/windows-build.md).
