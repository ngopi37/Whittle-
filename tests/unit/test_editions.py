import pytest

from core.editions import (
    FEATURE_MODEL_SIZE_LARGE,
    FEATURE_REMOTE_TRAINING,
    Edition,
    FeatureNotAvailable,
    resolve_entitlements,
)


def test_default_edition_is_free() -> None:
    ent = resolve_entitlements(env={}, license_path=None)
    assert ent.edition is Edition.FREE
    assert ent.has("local-pipeline")
    assert not ent.has(FEATURE_REMOTE_TRAINING)


def test_environment_override_selects_pro() -> None:
    ent = resolve_entitlements(env={"SG2_EDITION": "pro"})
    assert ent.edition is Edition.PRO
    assert ent.has(FEATURE_MODEL_SIZE_LARGE)


def test_license_file_selects_edition(tmp_path) -> None:
    license_file = tmp_path / "license.json"
    license_file.write_text('{"edition": "enterprise"}', encoding="utf-8")
    ent = resolve_entitlements(env={}, license_path=license_file)
    assert ent.edition is Edition.ENTERPRISE
    assert ent.has("sso")


def test_invalid_edition_name_falls_back_to_free() -> None:
    ent = resolve_entitlements(env={"SG2_EDITION": "platinum"})
    assert ent.edition is Edition.FREE


def test_require_raises_with_upgrade_hint() -> None:
    ent = resolve_entitlements(env={})
    with pytest.raises(FeatureNotAvailable, match="pro edition"):
        ent.require(FEATURE_REMOTE_TRAINING)
