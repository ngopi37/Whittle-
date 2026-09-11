"""Local dataset ingestion.

Imported content is treated strictly as data: it is parsed and normalized, never
executed. Every source file's path, size, modification time, and line count are
recorded as local provenance. No network access.

Exact-duplicate documents are dropped by default (``dedup``), and an optional
deterministic train/val split (``val_ratio``) can be requested — the split is a
stable hash of each document's text, so it is reproducible across runs and
independent of source file order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from core.pipeline.base import Stage, StageError
from core.pipeline.context import RunContext
from core.safety.paths import safe_resolve
from schemas.pipeline import ArtifactKind, StagePlan, StageResult

_TEXT_SUFFIXES = {".txt", ".text"}
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_PARQUET_SUFFIXES = {".parquet"}
_MAX_LINE_BYTES = 5_000_000


@dataclass
class _Document:
    text: str
    source: str


class DataIngestStage(Stage):
    """Normalize local text/JSONL/Parquet files into deduped, optionally split datasets."""

    name = "data-ingest"

    def plan(self, ctx: RunContext) -> StagePlan:
        sources = _requested_sources(ctx)
        approx_bytes = 0.0
        for source in sources:
            try:
                approx_bytes += source.stat().st_size
            except OSError:
                continue
        return StagePlan(
            stage=self.name,
            implemented=True,
            required_feature=None,
            inputs=[],
            outputs=[ArtifactKind.DATASET],
            estimated_memory_gb=round(0.1 + approx_bytes / 2**30, 3),
            estimated_disk_gb=round(approx_bytes / 2**30, 3),
            notes=[f"{len(sources)} source file(s) queued." if sources else "No sources provided."],
        )

    def run(self, ctx: RunContext) -> StageResult:
        sources = _requested_sources(ctx)
        if not sources:
            raise StageError("data-ingest requires at least one --input file.")

        dedup = bool(ctx.params.get("dedup", True))
        raw_val_ratio = ctx.params.get("val_ratio", 0.0)
        val_ratio = float(raw_val_ratio) if isinstance(raw_val_ratio, (int, float)) else 0.0
        if not 0.0 <= val_ratio < 1.0:
            raise StageError(f"val_ratio must be in [0, 1), got {val_ratio}.")
        split_seed = str(ctx.params.get("split_seed", "0"))

        documents: list[_Document] = []
        seen: set[str] = set()
        duplicates = 0
        provenance: list[dict[str, Any]] = []
        for source in sources:
            file_documents, record = _ingest_file(source)
            for doc in file_documents:
                if dedup and doc.text in seen:
                    duplicates += 1
                    continue
                seen.add(doc.text)
                documents.append(doc)
            provenance.append(record)
            ctx.log(
                f"Ingested {record['documents']} document(s) from {source.name}.", stage=self.name
            )

        train_docs, val_docs = _split(documents, val_ratio, split_seed)

        dataset_path = safe_resolve(ctx.workspace, "dataset.jsonl")
        _write_documents(dataset_path, train_docs)
        artifacts = [
            ctx.record_artifact(
                dataset_path,
                kind=ArtifactKind.DATASET,
                stage=self.name,
                provenance={
                    "role": "train" if val_docs else "all",
                    "documents": len(train_docs),
                    "duplicates_removed": duplicates,
                    "sources": provenance,
                },
            )
        ]

        if val_docs:
            val_path = safe_resolve(ctx.workspace, "dataset.val.jsonl")
            _write_documents(val_path, val_docs)
            artifacts.append(
                ctx.record_artifact(
                    val_path,
                    kind=ArtifactKind.DATASET,
                    stage=self.name,
                    provenance={
                        "role": "val",
                        "documents": len(val_docs),
                        "val_ratio": val_ratio,
                        "sources": provenance,
                    },
                )
            )

        meta_path = safe_resolve(ctx.workspace, "dataset.meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(tz=UTC).isoformat(),
                    "train_documents": len(train_docs),
                    "val_documents": len(val_docs),
                    "duplicates_removed": duplicates,
                    "sources": provenance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        metrics = {
            "documents": float(len(train_docs)),
            "sources": float(len(sources)),
            "duplicates_removed": float(duplicates),
        }
        if val_docs:
            metrics["val_documents"] = float(len(val_docs))
        return StageResult(
            stage=self.name,
            status="succeeded",
            artifacts=artifacts,
            metrics=metrics,
            message=(
                f"Normalized {len(train_docs)} train + {len(val_docs)} val document(s) "
                f"from {len(sources)} file(s), dropped {duplicates} duplicate(s)."
            ),
        )


def _requested_sources(ctx: RunContext) -> list[Path]:
    """Resolve and validate the ``inputs`` param into existing files."""
    raw = ctx.params.get("inputs", [])
    if isinstance(raw, (str, Path)):
        raw = [raw]
    if not isinstance(raw, list):
        raise StageError("data-ingest 'inputs' must be a list of file paths.")
    sources: list[Path] = []
    for item in raw:
        path = Path(str(item)).expanduser().resolve()
        if not path.is_file():
            raise StageError(f"Input is not a file: {path}")
        suffix = path.suffix.lower()
        if suffix not in _TEXT_SUFFIXES | _JSONL_SUFFIXES | _PARQUET_SUFFIXES:
            raise StageError(
                f"Unsupported input type {path.suffix!r}; expected .txt, .jsonl, or .parquet."
            )
        sources.append(path)
    return sources


def _split(
    documents: list[_Document], val_ratio: float, split_seed: str
) -> tuple[list[_Document], list[_Document]]:
    """Deterministically partition documents by a stable hash of their text.

    Hash-based (rather than positional or shuffled) so the split is reproducible
    regardless of source file order and stable if more documents are added later.
    """
    if val_ratio <= 0.0:
        return documents, []
    train: list[_Document] = []
    val: list[_Document] = []
    for doc in documents:
        digest = hashlib.sha256(f"{split_seed}:{doc.text}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        (val if bucket < val_ratio else train).append(doc)
    return train, val


def _write_documents(path: Path, documents: list[_Document]) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for doc in documents:
            sink.write(json.dumps({"text": doc.text}, ensure_ascii=False) + "\n")


def _ingest_file(source: Path) -> tuple[list[_Document], dict[str, Any]]:
    """Parse one source file into documents, returning them plus its provenance."""
    stat = source.stat()
    suffix = source.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        documents = _ingest_parquet(source)
        record_format = "parquet"
        lines = len(documents)
    else:
        is_jsonl = suffix in _JSONL_SUFFIXES
        documents, lines = _ingest_text_or_jsonl(source, is_jsonl)
        record_format = "jsonl" if is_jsonl else "text"
    return documents, {
        "path": str(source),
        "bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "lines": lines,
        "documents": len(documents),
        "format": record_format,
    }


def _ingest_text_or_jsonl(source: Path, is_jsonl: bool) -> tuple[list[_Document], int]:
    documents: list[_Document] = []
    lines = 0
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines += 1
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped.encode("utf-8")) > _MAX_LINE_BYTES:
                raise StageError(f"{source.name}:{lines} exceeds the maximum line size.")
            text = _normalize_line(stripped, is_jsonl, source.name, lines)
            documents.append(_Document(text=text, source=source.name))
    return documents, lines


def _ingest_parquet(source: Path) -> list[_Document]:
    """Read a ``text`` string column from a Parquet file, batch by batch."""
    parquet_file = pq.ParquetFile(source)
    columns = parquet_file.schema_arrow.names
    if "text" not in columns:
        raise StageError(f"{source.name} has no 'text' column; found {columns}.")
    documents: list[_Document] = []
    for batch in parquet_file.iter_batches(columns=["text"]):
        for value in batch.column("text").to_pylist():
            if isinstance(value, str) and value.strip():
                documents.append(_Document(text=value.strip(), source=source.name))
    return documents


def _normalize_line(
    stripped: str, is_jsonl: bool, source_name: str, line_number: int
) -> str:
    """Return the record's text; JSONL content is parsed as data only."""
    if not is_jsonl:
        return stripped
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StageError(f"{source_name}:{line_number} is not valid JSON: {exc}") from None
    if isinstance(parsed, dict):
        text = parsed.get("text")
        if isinstance(text, str):
            return text
    if isinstance(parsed, str):
        return parsed
    raise StageError(
        f"{source_name}:{line_number} must be a string or an object with a 'text' field."
    )
