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


def test_environment_variable_cannot_grant_a_paid_edition(tmp_path) -> None:
    # SG2_EDITION is a local "force free" convenience; it never grants pro/enterprise
    # on its own — only a validly signed license file does (see below).
    ent = resolve_entitlements(env={"SG2_EDITION": "pro"}, license_path=tmp_path / "none.json")
    assert ent.edition is Edition.FREE


def test_environment_variable_can_force_free_over_a_valid_license(sign_test_license) -> None:
    license_path = sign_test_license("enterprise")
    ent = resolve_entitlements(env={"SG2_EDITION": "free"}, license_path=license_path)
    assert ent.edition is Edition.FREE


def test_signed_license_file_selects_edition(sign_test_license) -> None:
    license_path = sign_test_license("enterprise")
    ent = resolve_entitlements(env={}, license_path=license_path)
    assert ent.edition is Edition.ENTERPRISE
    assert ent.has("sso")
    assert ent.has(FEATURE_MODEL_SIZE_LARGE)


def test_unsigned_license_file_is_ignored(tmp_path) -> None:
    license_file = tmp_path / "license.json"
    license_file.write_text('{"edition": "enterprise"}', encoding="utf-8")
    ent = resolve_entitlements(env={}, license_path=license_file)
    assert ent.edition is Edition.FREE


def test_tampered_license_file_is_ignored(sign_test_license) -> None:
    license_path = sign_test_license("enterprise")
    import json

    data = json.loads(license_path.read_text(encoding="utf-8"))
    data["edition"] = "pro"  # change a signed field after the fact
    license_path.write_text(json.dumps(data), encoding="utf-8")

    ent = resolve_entitlements(env={}, license_path=license_path)
    assert ent.edition is Edition.FREE


def test_expired_license_file_is_ignored(sign_test_license) -> None:
    license_path = sign_test_license("enterprise", expires_at="2000-01-01T00:00:00+00:00")
    ent = resolve_entitlements(env={}, license_path=license_path)
    assert ent.edition is Edition.FREE


def test_unexpired_license_file_is_honored(sign_test_license) -> None:
    license_path = sign_test_license("pro", expires_at="2999-01-01T00:00:00+00:00")
    ent = resolve_entitlements(env={}, license_path=license_path)
    assert ent.edition is Edition.PRO


def test_invalid_edition_name_falls_back_to_free() -> None:
    ent = resolve_entitlements(env={"SG2_EDITION": "platinum"})
    assert ent.edition is Edition.FREE


def test_require_raises_with_upgrade_hint() -> None:
    ent = resolve_entitlements(env={})
    with pytest.raises(FeatureNotAvailable, match="pro edition"):
        ent.require(FEATURE_REMOTE_TRAINING)
