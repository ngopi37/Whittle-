"""Out-of-process training worker for the ``pretrain`` stage.

Invoked as ``python -m core.pipeline.workers.train_worker <jobspec.json>``. Reads a
``JobSpec`` from that file, trains ``core.training.model.LlamaModel`` on the
tokenized dataset, and streams ``JobEvent``s (``progress`` / ``metric`` / ``log`` /
``artifact`` / ``done`` / ``error``) to stdout, one JSON object per line, so the
parent process (``PretrainStage``, which never imports ``torch`` itself) can render
progress without sharing memory with the worker.

Checkpoints live at ``params["checkpoint_dir"]/checkpoint_last.pt`` — a project-level
path, not the run's own workspace — so killing the process and re-running the same
``pipeline run ... --stage pretrain`` resumes from the last saved step instead of
starting over.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from schemas.pipeline import JobEvent, JobSpec

Emit = Callable[[str, dict[str, Any]], None]


def _stdout_emit(kind: str, payload: dict[str, Any]) -> None:
    event = JobEvent(schema_version=1, at=datetime.now(tz=UTC), kind=kind, payload=payload)  # type: ignore[arg-type]
    sys.stdout.write(event.model_dump_json() + "\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: train_worker.py <jobspec.json>", file=sys.stderr)
        return 2
    job = JobSpec.model_validate_json(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        run(job, emit=_stdout_emit)
    except Exception as exc:
        _stdout_emit("error", {"message": str(exc)})
        return 1
    return 0


def run(job: JobSpec, *, emit: Emit) -> None:
    """Train (or resume training) per ``job.params`` and stream progress via ``emit``.

    Kept separate from ``main`` so tests can call it directly, in-process, without
    spawning a subprocess.
    """
    import torch

    from core.training.data import tokenize_to_blocks
    from core.training.model import LlamaConfig, LlamaModel

    params = job.params
    dataset_path = Path(str(params["dataset_path"]))
    model_init_path = Path(str(params["model_init_path"]))
    tokenizer_path = Path(str(params["tokenizer_path"]))
    checkpoint_dir = Path(str(params["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_last.pt"

    block_size = int(params.get("block_size", 128))
    batch_size = max(1, int(params.get("batch_size", 1)))
    learning_rate = float(params.get("learning_rate", 3e-4))
    grad_accum = max(1, int(params.get("gradient_accumulation_steps", 1)))
    mixed_precision = bool(params.get("mixed_precision", False))
    max_steps = int(params.get("max_steps", 200))
    checkpoint_every = max(1, int(params.get("checkpoint_every", 50)))
    seed = int(params.get("seed", 1234))

    init_state = torch.load(model_init_path, map_location="cpu", weights_only=False)
    arch = LlamaConfig(**init_state["config"])
    model = LlamaModel(arch)
    model.load_state_dict(init_state["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_step = 0
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_step = ckpt["step"]
        emit("log", {"message": f"Resuming from checkpoint at step {start_step}."})
    else:
        torch.manual_seed(seed)

    blocks = tokenize_to_blocks(dataset_path, tokenizer_path, block_size)
    if not blocks:
        raise ValueError(
            f"Dataset has fewer than {block_size + 1} tokens; reduce block_size or add data."
        )
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed)).tolist()

    autocast = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if mixed_precision
        else nullcontext()
    )

    model.train()
    last_loss = float("nan")
    step = start_step
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(grad_accum):
            batch_ids = [order[(step * batch_size + i) % len(order)] for i in range(batch_size)]
            inputs = torch.stack([blocks[i][:-1] for i in batch_ids])
            targets = torch.stack([blocks[i][1:] for i in batch_ids])
            with autocast:
                _, loss = model(inputs, labels=targets)
            assert loss is not None
            (loss / grad_accum).backward()
            accumulated += loss.item() / grad_accum
        optimizer.step()
        last_loss = accumulated
        step += 1
        emit("progress", {"step": step, "max_steps": max_steps})
        emit("metric", {"step": step, "loss": last_loss})

        if step % checkpoint_every == 0 or step == max_steps:
            _save_checkpoint(checkpoint_path, model, optimizer, step, arch.__dict__)
            emit("artifact", {"path": str(checkpoint_path), "step": step})

    _save_checkpoint(checkpoint_path, model, optimizer, step, arch.__dict__)
    emit(
        "done",
        {"final_loss": last_loss, "steps": step, "checkpoint_path": str(checkpoint_path)},
    )


def _save_checkpoint(
    path: Path, model: Any, optimizer: Any, step: int, config: dict[str, Any]
) -> None:
    """Save model/optimizer state plus the architecture config, so downstream stages
    (``evaluate``, ``quantize``) can rebuild the model from the checkpoint alone."""
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 1,
            "step": step,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
