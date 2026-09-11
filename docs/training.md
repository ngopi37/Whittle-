# Training

Shipped — roadmap **P3** ([product/roadmap.md](product/roadmap.md)). `model-init`
and `pretrain` are both implemented.

## Architecture (`core.training.model.LlamaModel`)

A small Llama-style decoder-only transformer: RMSNorm, rotary position embeddings,
SwiGLU feed-forward, no biases, untied embeddings. Chosen deliberately so a trained
checkpoint maps onto GGUF's `llama` tensor layout by renaming, not re-deriving —
see [runtime-targets.md](runtime-targets.md). `model-init` sizes the embedding to
the tokenizer's *actual* vocabulary (from the `tokenizer` artifact's
`vocab_size_actual`), which can be smaller than the catalog's target on a small
corpus.

## `model-init` (in-process)

Instantiates the architecture and saves randomly-initialized weights as the
`model_init` artifact. Runs in-process (unlike `pretrain`) — even at 500M
parameters, initializing weights is fast enough not to need worker isolation.
`torch` is imported lazily inside `run()`, so importing the module (which the
pipeline registry does on every CLI/desktop invocation) never pays torch's import
cost.

## `pretrain` (out-of-process worker)

- Runs in an **out-of-process worker**: `PretrainStage` never imports `torch` —
  it shells out to `python -m core.pipeline.workers.train_worker <jobspec.json>`
  and reads the worker's streamed `JobEvent`s (`progress` / `metric` / `log` /
  `artifact` / `done` / `error`) from its stdout, one JSON object per line. The
  protocol is `JobSpec` v1 (`schemas/pipeline.py`); bump `schema_version` on any
  breaking change.
- **Checkpoints are project-scoped, not run-scoped:** they live at
  `.sg2/checkpoints/<project_name>/pretrain/checkpoint_last.pt` (under `cwd`, like
  `RunStore`'s default root), not inside the run's own workspace. That's what
  makes resume work across separate CLI invocations, each of which creates a new
  run directory — killing the process and re-running
  `pipeline run ... --stage pretrain` on the same project resumes from the last
  saved step. The checkpoint carries the architecture config too, so `evaluate`
  and `quantize` can rebuild the model from the checkpoint alone.
- **Verified, not just claimed:** `tests/unit/test_train_worker.py` calls the
  worker's `run()` directly (fast, in-process, no subprocess) and asserts loss
  trends down over real training steps, and that a second call resumes from the
  first's saved step rather than restarting at zero.
  `tests/unit/test_pretrain.py` additionally exercises the real worker subprocess
  end to end.
- Local training of models ≤ 100M is free. Remote / multi-GPU training is part of
  the Pro/Enterprise roadmap, gated by the `remote-training` feature (`--remote`).
- Resource feasibility comes from `core.sizing` using the project's own
  `TrainingConfig`; run `project refresh` after changing training knobs.

## Params (`pipeline run --stage pretrain`)

`--max-steps`, `--block-size`, `--checkpoint-every`, `--seed`. Batch size, learning
rate, gradient accumulation, and mixed precision come from the project's
`TrainingConfig` (`project create --batch-size` / `--mixed-precision` at creation
time).
