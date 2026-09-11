"""Local, filesystem-backed run tracking.

Layout::

    <root>/<run_id>/
        run.json          # RunRecord snapshot
        events.jsonl       # append-only RunEvent log
        artifacts/         # stage outputs
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from pathlib import Path

from schemas.pipeline import Artifact, ArtifactKind, RunEvent, RunRecord, RunStatus, StageName

DEFAULT_RUNS_ROOT = Path.cwd() / ".sg2" / "runs"
_HASH_CHUNK = 1024 * 1024


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _new_run_id() -> str:
    return f"{_now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class RunHandle:
    """A single run directory with helpers to append events and record artifacts."""

    def __init__(self, directory: Path, record: RunRecord) -> None:
        self.dir = directory
        self.record = record

    @property
    def artifacts_dir(self) -> Path:
        path = self.dir / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    @property
    def record_path(self) -> Path:
        return self.dir / "run.json"

    def save(self) -> None:
        self.record_path.write_text(self.record.model_dump_json(indent=2), encoding="utf-8")

    def append_event(
        self, message: str, *, stage: StageName | None = None, level: str = "info"
    ) -> None:
        event = RunEvent(at=_now(), stage=stage, level=level, message=message)  # type: ignore[arg-type]
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def set_status(self, status: RunStatus) -> None:
        self.record.status = status
        self.save()

    def set_stage_status(self, stage: StageName, status: RunStatus) -> None:
        self.record.stages[stage] = status
        self.save()

    def record_artifact(
        self,
        path: Path,
        *,
        kind: ArtifactKind,
        stage: StageName,
        provenance: dict[str, object] | None = None,
    ) -> Artifact:
        sha256, size = _sha256_and_size(path)
        artifact = Artifact(
            id=f"{stage}-{kind.value}-{secrets.token_hex(4)}",
            kind=kind,
            path=str(path),
            sha256=sha256,
            bytes=size,
            produced_by_stage=stage,
            provenance=provenance or {},
        )
        self.record.artifacts.append(artifact)
        self.save()
        return artifact


class RunStore:
    """Create, load, and list runs under a local root directory."""

    def __init__(self, root: Path = DEFAULT_RUNS_ROOT) -> None:
        self.root = root

    def create_run(self, *, project_name: str, model_size: str) -> RunHandle:
        run_id = _new_run_id()
        directory = self.root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            id=run_id,
            project_name=project_name,
            model_size=model_size,
            created_at=_now(),
            status="pending",
        )
        handle = RunHandle(directory, record)
        handle.save()
        handle.append_event(f"Run {run_id} created for project '{project_name}'.")
        return handle

    def load_run(self, run_id: str) -> RunHandle:
        directory = self.root / run_id
        record = RunRecord.model_validate_json((directory / "run.json").read_text(encoding="utf-8"))
        return RunHandle(directory, record)

    def list_runs(self) -> list[RunRecord]:
        if not self.root.is_dir():
            return []
        records: list[RunRecord] = []
        for entry in sorted(self.root.iterdir()):
            record_file = entry / "run.json"
            if record_file.is_file():
                records.append(
                    RunRecord.model_validate_json(record_file.read_text(encoding="utf-8"))
                )
        return records
