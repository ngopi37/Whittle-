"""Shared test fixtures."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import core.licensing as licensing


@pytest.fixture
def sign_test_license(tmp_path, monkeypatch):
    """Return a factory that writes a validly-signed license.json under a throwaway
    keypair, monkeypatching ``core.licensing.PUBLIC_KEY_B64`` so verification uses it.

    Tests never touch the real production private key — it doesn't exist anywhere
    in this repo.
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_b64 = base64.b64encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode()
    public_b64 = base64.b64encode(
        public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode()
    monkeypatch.setattr(licensing, "PUBLIC_KEY_B64", public_b64)

    def _make(edition: str = "pro", *, path: Path | None = None, **fields: Any) -> Path:
        payload = {"edition": edition, **fields}
        signed = licensing.sign_license(payload, private_b64)
        license_path = path or (tmp_path / "license.json")
        license_path.write_text(json.dumps(signed), encoding="utf-8")
        return license_path

    return _make
