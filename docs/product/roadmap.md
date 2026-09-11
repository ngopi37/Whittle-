# Product Roadmap

SG2 automates building a small model that runs on the device in front of you:
profile the machine, prepare data, train, evaluate, quantize, package for a target
runtime, and smoke-test it — all locally, offline, on Windows first.

Each phase has explicit entry and exit criteria. A phase ships only when its exit
criteria are met and the public contracts it introduced are stable.

| Phase | Scope | Edition | Status |
|------|-------|---------|--------|
| **P1** | Hardware Intelligence | free | **shipped** |
| **P2** | Data & Tokenizer | free | **shipped** |
| **P3** | Training (local worker) | free | **shipped** |
| **P4** | Evaluation & Quantization | free | **shipped (GGUF only)** |
| **P5** | Packaging & On-Device Test | free | **shipped (GGUF only)** |
| **P6** | Model Registry & Fleet Deploy | paid | planned |
| **P7** | Teams & Governance | paid | planned |

P4/P5 ship the full local pipeline for **GGUF/llama.cpp only**. ExecuTorch and
TFLite/Core ML are still deferred — see [runtime-targets.md](../runtime-targets.md)
for why (different, heavy toolchains; Core ML can't be verified off macOS) — and
are not silently missing: `pipeline plan` and this doc call them out explicitly.

## P1 — Hardware Intelligence (shipped)

Measured device profiling, conservative resource sizing driven by the project's own
model and training config, disk gating, YAML model catalog (20M–500M), local project
persistence, CLI + desktop shell, and the pipeline/edition foundation.

- **Exit criteria (met):** profiler normalizes CPU/RAM/GPU/storage on Windows and
  degrades cleanly elsewhere; sizing is an explainable function of inputs; `pytest`,
  `ruff`, and `mypy --strict` are clean; `pipeline plan` shows the whole build.

## P2 — Data & Tokenizer (shipped)

- `data-ingest`: `.jsonl` / `.txt` / `.parquet` sources, exact-duplicate dedup (on
  by default), a deterministic hash-based train/val split (`--val-ratio`), local
  provenance per source.
- `tokenizer-train`: BPE (byte-level) or unigram via the `tokenizers` library;
  vocab size defaults to the project's `ModelConfig`; records algorithm/vocab/
  special-token provenance on the `tokenizer` artifact.
- **Exit criteria (met):** a project goes from raw `.txt`/`.jsonl`/`.parquet` files
  to a deduped, optionally split corpus and a tokenizer artifact with one
  `pipeline run`. Archive ingest and dataset cards/license capture (noted in
  `docs/datasets.md`) were never part of this phase's exit criteria and remain
  future work.

## P3 — Training (local worker) (shipped)

- `model-init` builds a Llama-style decoder (RMSNorm, RoPE, SwiGLU, no biases —
  chosen so it maps directly onto GGUF's `llama` tensor layout) from
  `ModelConfig`, sized to the tokenizer's *actual* vocab.
- `pretrain` runs in an **out-of-process worker**
  (`core.pipeline.workers.train_worker`) driven by `JobSpec` v1
  (`schemas/pipeline.py`); `PretrainStage` itself never imports `torch`.
- Checkpoints live under a project-scoped directory (`.sg2/checkpoints/<project>/
  pretrain/`), not the run's own workspace, so a killed-and-restarted
  `pipeline run ... --stage pretrain` resumes from the last saved step instead of
  retraining from scratch.
- **Exit criteria (met):** verified by `tests/unit/test_train_worker.py` (loss
  decreases over real training steps; a second run resumes from the first's
  checkpoint step rather than restarting) and `tests/unit/test_pretrain.py` (the
  real worker subprocess, not a direct call).

## P4 — Evaluation & Quantization (shipped, GGUF only)

- `evaluate`: held-out perplexity plus next-token top-1 accuracy on the val split;
  `eval_report` artifact. Runs in-process (a single forward pass is fast even at
  100M parameters).
- `quantize`: GGUF export (`fp16` / `q8_0` / `q4_0`) via `core.runtime.gguf_export`,
  using the `gguf` package's own reference quantization kernels — the same ones
  llama.cpp's `convert_hf_to_gguf.py` uses — not a hand-rolled approximation.
- **Exit criteria (met, GGUF only):** verified in `tests/unit/test_gguf_pipeline.py`
  by actually loading the exported file with real `llama.cpp`
  (`llama-cpp-python`) and generating tokens — not just that the writer ran
  without raising. ExecuTorch and TFLite/Core ML quantization remain deferred.

## P5 — Packaging & On-Device Test (shipped, GGUF only)

- `package`: zips the `.gguf` (which embeds its own tokenizer — llama.cpp reads it
  straight out of the file) with a `manifest.json` (model size, quant type,
  tokenizer hash, source run id, sizing recommendation).
- `on-device-smoke-test`: extracts the package, loads it with real `llama.cpp`,
  generates a configurable prompt/token count, and records tokens/sec and process
  RSS.
- **Exit criteria (met, GGUF only):** the same end-to-end test proves a real
  runtime accepts the packaged file. ExecuTorch and TFLite/Core ML packaging and
  their signed/versioned variant remain deferred.

## P6 — Model Registry & Fleet Deploy (paid)

Hosted (or self-hosted) registry for versioned packages, and OTA delivery to a fleet
of devices with staged rollout and rollback. Gated by `model-registry` /
`fleet-deploy`.

## P7 — Teams & Governance (paid)

Shared projects with RBAC, SSO/SCIM, a policy engine (allowed data sources, model
sizes, targets), and an append-only audit sink. Gated by `teams`, `sso`,
`policy-engine`, `audit-sink`.

See [editions.md](editions.md) for the free/paid boundary and
[../pipeline.md](../pipeline.md) for the stage architecture.
