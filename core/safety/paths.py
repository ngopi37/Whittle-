"""Filesystem path safety.

All user-supplied paths that will be read or written under a controlled base
directory must go through :func:`safe_resolve`. Archive extraction is not implemented
yet; :func:`safe_extract` exists so the contract is visible and enforced when it lands.
"""

from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a candidate path escapes its permitted base directory."""


def safe_resolve(base: Path | str, candidate: Path | str) -> Path:
    """Resolve ``candidate`` and confirm it stays within ``base``.

    ``candidate`` may be relative (joined onto ``base``) or absolute (it must then
    already live under ``base``). Symlinks and ``..`` segments that would escape the
    base raise :class:`UnsafePathError`.
    """
    base_path = Path(base).resolve()
    raw = Path(candidate)
    joined = raw if raw.is_absolute() else base_path / raw
    resolved = joined.resolve()
    if resolved != base_path and base_path not in resolved.parents:
        raise UnsafePathError(f"{candidate!r} resolves outside {base_path}")
    return resolved


def safe_extract(archive: Path | str, dest: Path | str) -> None:
    """Extract an archive under ``dest`` without path traversal.

    Not implemented in the current phase. When added it must validate every member
    name with :func:`safe_resolve`, refuse absolute members and symlink members, and
    never execute archive content.
    """
    raise NotImplementedError(
        "Archive extraction is deferred; see docs/security.md for the required contract."
    )
