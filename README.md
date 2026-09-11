# Whittle

License: Apache 2.0

Open-source, on-device pipeline for building small language models that run on the machine in front of you.

Whittle profiles your hardware, then takes it from raw text files all the way to a quantized, packaged, on-device model — tokenizer training, model architecture, pretraining, evaluation, GGUF quantization, packaging, and a real on-device smoke test — entirely offline, with no cloud training service and no API key. Everything runs locally; nothing you feed it is ever uploaded anywhere.

This repository is the Free edition: the CLI (`apps/cli`), the desktop shell (`apps/desktop`), and the full local pipeline engine (`core/`) that actually profiles your hardware, trains, and packages a model — the same components used across every Whittle edition, published together so the piece that touches your data and your machine is just as inspectable as the plan behind it. Licensed under Apache 2.0: use it, modify it, self-host it, or build on it, commercially or not — see License below.

## Features

- **Hardware profiling & sizing** — measures CPU, RAM, GPU, and storage on Windows (with graceful degradation elsewhere) and recommends a model size and training configuration that will actually fit, before you spend an hour training something that won't
- **Local dataset ingest** — `.jsonl`, `.txt`, and `.parquet` sources; exact-duplicate removal (on by default); a deterministic, hash-based train/val split (`--val-ratio`) that's reproducible regardless of source-file order; per-source provenance recorded alongside every dataset artifact
- **Tokenizer training** — byte-level BPE or unigram, via the `tokenizers` library, sized to the project's model config
- **Model architecture** — a small Llama-style decoder (RMSNorm, rotary position embeddings, SwiGLU feed-forward) sized from a YAML catalog (20M–500M parameters), deliberately chosen so it maps directly onto GGUF's tensor layout for export
- **Pretraining** — runs in an out-of-process worker, CPU-only; checkpoints live in a project-scoped directory so killing the process and re-running training resumes from the last saved step instead of starting over
- **Evaluation** — held-out perplexity and next-token top-1 accuracy on the val split
- **GGUF quantization** — `fp16`, `q8_0` (default), or `q4_0`, using the `gguf` package's own reference quantization kernels — the same ones llama.cpp's own conversion scripts use, not a hand-rolled approximation
- **Packaging & on-device smoke test** — bundles the quantized model plus a manifest into one `.zip`, then actually loads it with real `llama.cpp` (via `llama-cpp-python`) and generates tokens — proof the file works on a real runtime, not just that the exporter didn't crash
- **Local run tracking** — every pipeline run gets its own directory (`.sg2/runs/<run-id>/`) with an append-only event log and every produced artifact recorded with a sha256 hash
- **Offline license verification** — no account, no network call, ever

## Installation

Requires Python 3.12+. `torch` and `llama-cpp-python` are heavy (hundreds of MB) — use the CPU-only torch wheel unless you specifically want CUDA:

```powershell
git clone https://github.com/ngopi37/Whittle-.git
cd Whittle-
python -m pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

There's no pre-built installer yet — see [Limitations](#limitations) below. Everything runs from source on Windows, Linux, and macOS (the hardware profiler degrades gracefully off Windows; the local pipeline itself has no OS-specific code).

## Verify installation

```powershell
python -m apps.cli.main hardware    # should print your measured CPU/RAM/GPU/storage
python -m apps.cli.main edition     # should print {"edition": "free", ...}
pytest                              # should be all green
```

## Using it

The desktop shell (`python -m apps.desktop.main`, Tkinter, no network) opens immediately and detects your hardware on a worker thread — a good place to start if you just want to see what your machine can train.

For the full build, the CLI runs one stage at a time or the whole pipeline in a single command:

```powershell
# Profile your hardware and see what fits.
python -m apps.cli.main hardware
python -m apps.cli.main recommend --batch-size 8

# Create a project sized to your machine.
python -m apps.cli.main project create my-model --model 20M --with-hardware --output project.json

# See the whole build, per-stage, before running anything.
python -m apps.cli.main pipeline plan project.json

# Raw text in, a working GGUF package out — one command.
python -m apps.cli.main pipeline run project.json `
  --stage data-ingest --input corpus.jsonl --val-ratio 0.1 `
  --stage tokenizer-train --vocab-size 4000 `
  --stage model-init `
  --stage pretrain --max-steps 200 `
  --stage evaluate `
  --stage quantize --quant-type q8_0 `
  --stage package `
  --stage on-device-smoke-test --prompt "Hello" --max-tokens 32
```

Each `--stage` shares one run, so later stages see earlier stages' artifacts automatically (the trained tokenizer, the checkpoint, the quantized model, ...). Run stages separately across multiple commands and they each get their own run directory — `pretrain` is the one exception: its checkpoints live under `.sg2/checkpoints/<project-name>/pretrain/`, not the run directory, specifically so re-running `--stage pretrain` after killing the process resumes instead of retraining from scratch.

See [docs/pipeline.md](docs/pipeline.md) for the stage architecture and [docs/training.md](docs/training.md) / [docs/runtime-targets.md](docs/runtime-targets.md) for what each stage actually does.

## Editions

Local pipeline features are free, always. Scale, team, and governance capabilities (remote training, a model registry, fleet deployment, teams, SSO, a policy engine, an audit sink) are part of the Pro/Enterprise roadmap — see [docs/product/roadmap.md](docs/product/roadmap.md). Edition checks are entirely offline and cryptographically verified — no account, no network call. Contact info@sg2technologies.com for a license.

```powershell
python -m apps.cli.main edition
```

## Limitations

These are the known gaps in what's implemented today. Listed here rather than left to be discovered.

**Runtime targets** — only **GGUF/llama.cpp** is implemented. ExecuTorch and TFLite/Core ML are on the roadmap but not started: no export code, no tests, nothing to point at. Core ML in particular can't be verified without macOS, which this project doesn't have access to yet — see [docs/runtime-targets.md](docs/runtime-targets.md).

**Quantization** — `fp16`, `q8_0`, and `q4_0` only. The "k-quant" superblock formats (`Q5_K_M`, `Q4_K_M`) are supported by the underlying `gguf` library already, so adding them is mostly wiring, not new quantization math — just not done yet.

**Dataset ingest** — no archive ingest (`.zip`, etc.), no dataset cards or license capture. `.parquet`/`.jsonl`/`.txt` with a `text` field/column is all that's read.

**Training** — CPU-only; no GPU acceleration path exists yet (remote/multi-GPU training is a roadmap item with no implementation). No distributed or multi-node training.

**Packaging** — no PyInstaller build yet; running the desktop shell or CLI requires a Python install and the full dependency set (`torch`, `llama-cpp-python`, `gguf`, `tokenizers`, `pyarrow`). No code signing (irrelevant until a build exists).

## Security & privacy

- Everything runs on-device — no training data, model weights, or telemetry is ever sent anywhere. There is no server component and no phone-home.
- Edition/license resolution is entirely offline: an environment variable or a local file, never a network call.
- Local run history (`.sg2/runs/`) and pretraining checkpoints (`.sg2/checkpoints/`) are plain local files; delete them like any other local directory.
- Dataset content is always treated as data, never executed — JSONL/Parquet is parsed, not evaluated, and file paths go through `core.safety.paths.safe_resolve` to reject path traversal.
- No account, sign-in, or license is required to build, run, or use the Free edition.

## Enterprise & Pro editions

Whittle's local pipeline — everything in this repository — is free forever. Need centralized model management across a fleet, remote/multi-GPU training, team collaboration with RBAC, SSO/SCIM, a policy engine, or an audit sink? Whittle Enterprise builds on this same local engine with fleet management, a shared model registry, and governance tooling for organizations that need it.

Enterprise is a separate, proprietary product. Contact us to talk about your requirements.

→ https://sg2technologies.com

## Contact

info@sg2technologies.com

## License

Copyright © 2026 SG2 Technologies.

Licensed under the Apache License, Version 2.0. You may use, modify, and redistribute this code — including commercially — under the terms of that license. See [LICENSE](LICENSE).

## Author

Gopi Narayanaswamy — [github.com/ngopi37](https://github.com/ngopi37)
