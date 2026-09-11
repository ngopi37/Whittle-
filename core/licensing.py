"""Offline Ed25519 signing and verification for ``license.json``.

Verification is public — this module ships the public key, which is exactly how
public-key cryptography is supposed to work. Producing a signature this module
accepts requires the matching *private* key, which is generated once, kept out of
this repository, and never touches this code. Nothing here, and no amount of
reading this source, lets anyone forge a license; only holding the private key
does.

Canonicalization: every field except ``signature``, serialized as
``json.dumps(payload, sort_keys=True, separators=(",", ":"))`` and UTF-8 encoded.
Signing and verification both use this, so changing any signed field (edition,
expiry, ...) after the fact invalidates the signature.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

#: Public key for verifying license.json signatures. Safe to publish.
PUBLIC_KEY_B64 = "LqatOYVzyQ/u8lA7LwxROigFV16wdndp3jSBYUmQ8Sw="

_SIGNATURE_FIELD = "signature"


def canonical_payload(license_data: dict[str, Any]) -> bytes:
    """Serialize every field except ``signature`` deterministically for signing."""
    payload = {k: v for k, v in license_data.items() if k != _SIGNATURE_FIELD}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_license(license_data: dict[str, Any], private_key_b64: str) -> dict[str, Any]:
    """Return ``license_data`` with a ``signature`` field added.

    ``private_key_b64`` is the raw 32-byte Ed25519 private key, base64-encoded.
    Only whoever issues licenses holds this; it is never checked into this repo.
    """
    private_key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    signature = private_key.sign(canonical_payload(license_data))
    return {**license_data, _SIGNATURE_FIELD: base64.b64encode(signature).decode("ascii")}


def verify_license(license_data: dict[str, Any]) -> bool:
    """Return whether ``license_data``'s ``signature`` is valid for its other fields."""
    raw_signature = license_data.get(_SIGNATURE_FIELD)
    if not isinstance(raw_signature, str):
        return False
    try:
        signature = base64.b64decode(raw_signature, validate=True)
    except (ValueError, TypeError):
        return False
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(PUBLIC_KEY_B64))
    try:
        public_key.verify(signature, canonical_payload(license_data))
    except InvalidSignature:
        return False
    return True


def is_expired(license_data: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Return whether ``license_data["expires_at"]`` (ISO 8601), if present, is past.

    A license with no ``expires_at`` never expires. An ``expires_at`` that fails to
    parse is treated as expired rather than ignored — a malformed date should not
    silently grant an unbounded license.
    """
    expires_at = license_data.get("expires_at")
    if expires_at is None:
        return False
    if not isinstance(expires_at, str):
        return True
    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return (now or datetime.now(tz=UTC)) > expiry
