from datetime import UTC, datetime

from core.licensing import is_expired, sign_license, verify_license


def test_sign_and_verify_roundtrip(sign_test_license) -> None:
    import json

    license_path = sign_test_license("pro", issued_to="customer@example.com")
    data = json.loads(license_path.read_text(encoding="utf-8"))
    assert verify_license(data)


def test_verify_rejects_missing_signature() -> None:
    assert not verify_license({"edition": "pro"})


def test_verify_rejects_tampered_payload(sign_test_license) -> None:
    import json

    license_path = sign_test_license("pro")
    data = json.loads(license_path.read_text(encoding="utf-8"))
    data["edition"] = "enterprise"
    assert not verify_license(data)


def test_verify_rejects_garbage_signature() -> None:
    assert not verify_license({"edition": "pro", "signature": "not-base64!!"})


def test_sign_license_requires_matching_private_key() -> None:
    # Signing with an unrelated (but validly-shaped) key must not verify against
    # the module's real PUBLIC_KEY_B64 — this test uses the actual production
    # public key, so it only passes if forging really is impossible without the
    # matching private key.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    unrelated_private_key = Ed25519PrivateKey.generate()
    import base64

    private_b64 = base64.b64encode(
        unrelated_private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode()
    forged = sign_license({"edition": "enterprise"}, private_b64)
    assert not verify_license(forged)


def test_is_expired_true_for_past_date() -> None:
    assert is_expired({"expires_at": "2000-01-01T00:00:00+00:00"})


def test_is_expired_false_for_future_date() -> None:
    assert not is_expired({"expires_at": "2999-01-01T00:00:00+00:00"})


def test_is_expired_false_when_absent() -> None:
    assert not is_expired({"edition": "pro"})


def test_is_expired_true_for_unparsable_date() -> None:
    assert is_expired({"expires_at": "not-a-date"})


def test_is_expired_respects_now_override() -> None:
    payload = {"expires_at": "2020-06-01T00:00:00+00:00"}
    assert not is_expired(payload, now=datetime(2020, 1, 1, tzinfo=UTC))
    assert is_expired(payload, now=datetime(2021, 1, 1, tzinfo=UTC))
