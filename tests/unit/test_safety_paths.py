import pytest

from core.safety.paths import UnsafePathError, safe_extract, safe_resolve


def test_safe_resolve_allows_paths_within_base(tmp_path) -> None:
    resolved = safe_resolve(tmp_path, "sub/file.txt")
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_safe_resolve_rejects_parent_traversal(tmp_path) -> None:
    with pytest.raises(UnsafePathError):
        safe_resolve(tmp_path / "base", "../escape.txt")


def test_safe_resolve_rejects_absolute_escape(tmp_path) -> None:
    with pytest.raises(UnsafePathError):
        safe_resolve(tmp_path / "base", tmp_path / "other" / "x.txt")


def test_safe_extract_is_not_implemented(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        safe_extract(tmp_path / "a.zip", tmp_path)
