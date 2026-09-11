"""Local tokenizer training.

Trains a byte-level BPE or unigram tokenizer on the ``dataset`` artifact produced by
``data-ingest``, entirely offline via the ``tokenizers`` library (no hub downloads,
no network access). The vocabulary size defaults to the project's own
``ModelConfig.vocabulary_size`` so the tokenizer stays consistent with the model it
will feed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.pipeline.stages._dataset import select_dataset_artifact
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult

_ALGORITHMS = ("bpe", "unigram")
_DEFAULT_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class TokenizerTrainStage(Stage):
    """Train a tokenizer on the ingested dataset and record a ``tokenizer`` artifact."""

    name = "tokenizer-train"

    def plan(self, ctx: RunContext) -> StagePlan:
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[ArtifactKind.DATASET],
            outputs=[ArtifactKind.TOKENIZER],
            estimated_memory_gb=0.5,
            estimated_disk_gb=0.05,
            notes=[
                "BPE (byte-level) or unigram, trained offline via the tokenizers library.",
                f"Vocabulary size defaults to the project model's "
                f"{ctx.project.model.vocabulary_size} unless overridden.",
            ],
        )

    def run(self, ctx: RunContext) -> StageResult:
        dataset = select_dataset_artifact(ctx, role="train")
        if dataset is None:
            raise StageError(
                "tokenizer-train requires a 'dataset' artifact; run data-ingest first."
            )
        dataset_path = Path(dataset.path)
        if not dataset_path.is_file():
            raise StageError(f"Dataset artifact is missing on disk: {dataset_path}")

        algorithm = str(ctx.params.get("algorithm", "bpe")).lower()
        if algorithm not in _ALGORITHMS:
            raise StageError(
                f"Unknown tokenizer algorithm {algorithm!r}; expected one of {_ALGORITHMS}."
            )
        vocab_size = _int_param(
            ctx.params.get("vocab_size"), default=ctx.project.model.vocabulary_size
        )
        if vocab_size <= 0:
            raise StageError(f"vocab_size must be positive, got {vocab_size}.")
        special_tokens = _list_param(
            ctx.params.get("special_tokens"), default=_DEFAULT_SPECIAL_TOKENS
        )

        documents = _count_documents(dataset_path)
        if documents == 0:
            raise StageError("tokenizer-train requires a non-empty dataset; got 0 documents.")

        tokenizer = _build_tokenizer(algorithm, vocab_size, special_tokens)
        trainer = _build_trainer(algorithm, vocab_size, special_tokens)
        tokenizer.train_from_iterator(_iter_documents(dataset_path), trainer=trainer)

        tokenizer_path = safe_resolve(ctx.workspace, "tokenizer.json")
        tokenizer.save(str(tokenizer_path))
        ctx.log(
            f"Trained {algorithm} tokenizer on {documents} document(s), "
            f"vocab size {tokenizer.get_vocab_size()}.",
            stage=self.name,
        )

        artifact = ctx.record_artifact(
            tokenizer_path,
            kind=ArtifactKind.TOKENIZER,
            stage=self.name,
            provenance={
                "algorithm": algorithm,
                "vocab_size_requested": vocab_size,
                "vocab_size_actual": tokenizer.get_vocab_size(),
                "special_tokens": special_tokens,
                "documents": documents,
                "dataset_artifact_id": dataset.id,
            },
        )
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=[artifact],
            metrics={
                "vocab_size": float(tokenizer.get_vocab_size()),
                "documents": float(documents),
            },
            message=f"Trained {algorithm} tokenizer, vocab size {tokenizer.get_vocab_size()}.",
        )


def _int_param(raw: object, *, default: int) -> int:
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    raise StageError(f"vocab_size must be an integer, got {raw!r}.")


def _list_param(raw: object, *, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return list(raw)
    raise StageError(f"special_tokens must be a list of strings, got {raw!r}.")


def _build_tokenizer(algorithm: str, vocab_size: int, special_tokens: list[str]) -> Tokenizer:
    if algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer


def _build_trainer(
    algorithm: str, vocab_size: int, special_tokens: list[str]
) -> trainers.Trainer:
    if algorithm == "bpe":
        return trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size, special_tokens=special_tokens, show_progress=False
        )
    return trainers.UnigramTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        show_progress=False,
    )


def _count_documents(dataset_path: Path) -> int:
    count = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def _iter_documents(dataset_path: Path) -> Iterator[str]:
    """Yield each document's ``text`` field, treating dataset content strictly as data."""
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record: Any = json.loads(line)
            text = record.get("text", "") if isinstance(record, dict) else ""
            if text:
                yield text
