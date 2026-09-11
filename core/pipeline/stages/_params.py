"""Typed extraction helpers for ``RunContext.params`` (``dict[str, object]``).

Every stage's ``params`` dict is untyped by design (it's the CLI/desktop's free-form
bridge into a stage), so pulling a concrete type out of it needs an explicit
``isinstance`` check for mypy's sake — these are that check, written once.
"""

from __future__ import annotations


def int_param(params: dict[str, object], key: str, default: int) -> int:
    """Return ``params[key]`` as an int, or ``default`` if absent, falsy, or not an int."""
    raw = params.get(key)
    return raw if isinstance(raw, int) and raw else default
