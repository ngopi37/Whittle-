"""Tokenizing a ``dataset.jsonl`` into fixed-length blocks for training/eval.

Shared by ``core.pipeline.workers.train_worker`` and
``core.pipeline.stages.evaluate`` so the two use exactly the same packing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer


def tokenize_to_blocks(
    dataset_path: Path, tokenizer_path: Path, block_size: int
) -> list[torch.Tensor]:
    """Concatenate every document's tokens (``<eos>``-separated) into ``block_size + 1`` chunks."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")

    ids: list[int] = []
    for text in iter_texts(dataset_path):
        ids.extend(tokenizer.encode(text).ids)
        if eos_id is not None:
            ids.append(eos_id)

    stride = block_size + 1
    return [
        torch.tensor(ids[i : i + stride], dtype=torch.long)
        for i in range(0, len(ids) - stride + 1, stride)
    ]


def iter_texts(dataset_path: Path) -> Iterator[str]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record: Any = json.loads(line)
            text = record.get("text", "") if isinstance(record, dict) else ""
            if text:
                yield text
