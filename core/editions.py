"""Offline edition and entitlement resolution.

The philosophy is *local free; scale, team, and governance paid*. Everything that runs
on a single machine is in the free edition. Paid editions unlock remote/distributed
training, a hosted model registry, fleet deployment, team collaboration, and enterprise
governance. Resolution is entirely offline: an environment variable or a local license
file, never a network call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

LICENSE_PATH = Path.home() / ".sg2" / "license.json"

# Feature identifiers. Keep these stable; they appear in license files and stage plans.
FEATURE_LOCAL_PIPELINE = "local-pipeline"
FEATURE_SINGLE_DEVICE = "single-device"
FEATURE_RUNTIME_GGUF = "runtime:gguf"
FEATURE_RUNTIME_EXECUTORCH = "runtime:executorch"
FEATURE_RUNTIME_TFLITE_COREML = "runtime:tflite-coreml"
FEATURE_MODEL_SIZE_SMALL = "model-size:<=100M"
FEATURE_MODEL_SIZE_LARGE = "model-size:>100M"
FEATURE_REMOTE_TRAINING = "remote-training"
FEATURE_MODEL_REGISTRY = "model-registry"
FEATURE_FLEET_DEPLOY = "fleet-deploy"
FEATURE_TEAMS = "teams"
FEATURE_SSO = "sso"
FEATURE_POLICY_ENGINE = "policy-engine"
FEATURE_AUDIT_SINK = "audit-sink"
FEATURE_ON_PREM_CONTROL_PLANE = "on-prem-control-plane"


class Edition(StrEnum):
    """Supported product editions in ascending capability order."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


_FREE_FEATURES = frozenset(
    {
        FEATURE_LOCAL_PIPELINE,
        FEATURE_SINGLE_DEVICE,
        FEATURE_RUNTIME_GGUF,
        FEATURE_RUNTIME_EXECUTORCH,
        FEATURE_RUNTIME_TFLITE_COREML,
        FEATURE_MODEL_SIZE_SMALL,
    }
)
_PRO_FEATURES = _FREE_FEATURES | {
    FEATURE_MODEL_SIZE_LARGE,
    FEATURE_REMOTE_TRAINING,
    FEATURE_MODEL_REGISTRY,
    FEATURE_FLEET_DEPLOY,
    FEATURE_TEAMS,
}
_ENTERPRISE_FEATURES = _PRO_FEATURES | {
    FEATURE_SSO,
    FEATURE_POLICY_ENGINE,
    FEATURE_AUDIT_SINK,
    FEATURE_ON_PREM_CONTROL_PLANE,
}

_FEATURES_BY_EDITION: dict[Edition, frozenset[str]] = {
    Edition.FREE: _FREE_FEATURES,
    Edition.PRO: frozenset(_PRO_FEATURES),
    Edition.ENTERPRISE: frozenset(_ENTERPRISE_FEATURES),
}

_UPGRADE_HINT = {
    Edition.FREE: "Upgrade to SG2 Pro to unlock this capability.",
    Edition.PRO: "This capability requires SG2 Enterprise.",
    Edition.ENTERPRISE: "This capability is not available in any current edition.",
}


class FeatureNotAvailable(RuntimeError):
    """Raised when a gated feature is used outside its edition."""

    def __init__(self, feature: str, edition: Edition) -> None:
        self.feature = feature
        self.edition = edition
        required = min(
            (ed for ed, feats in _FEATURES_BY_EDITION.items() if feature in feats),
            default=None,
            key=lambda ed: list(Edition).index(ed),
        )
        hint = _UPGRADE_HINT[edition]
        if required is not None:
            hint = f"'{feature}' requires the {required.value} edition. {hint}"
        super().__init__(hint)


@dataclass(frozen=True)
class Entitlements:
    """The resolved edition and the features it grants."""

    edition: Edition
    features: frozenset[str]

    def has(self, feature: str) -> bool:
        """Return whether a feature is granted."""
        return feature in self.features

    def require(self, feature: str) -> None:
        """Raise :class:`FeatureNotAvailable` unless the feature is granted."""
        if feature not in self.features:
            raise FeatureNotAvailable(feature, self.edition)


def _edition_from_name(value: str | None) -> Edition | None:
    """Parse an edition name case-insensitively."""
    if not value:
        return None
    try:
        return Edition(value.strip().lower())
    except ValueError:
        return None


def _edition_from_license(path: Path) -> Edition | None:
    """Read the edition from a local license file if present and well formed.

    Signature verification is intentionally deferred (TODO: verify an offline-issued
    signature before honoring a paid edition). No network access is ever performed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return _edition_from_name(raw.get("edition"))


def resolve_entitlements(
    *, env: dict[str, str] | None = None, license_path: Path | None = None
) -> Entitlements:
    """Resolve the active entitlements from the environment or a local license file."""
    environment = os.environ if env is None else env
    edition = _edition_from_name(environment.get("SG2_EDITION"))
    if edition is None:
        edition = _edition_from_license(license_path or LICENSE_PATH)
    if edition is None:
        edition = Edition.FREE
    return Entitlements(edition=edition, features=_FEATURES_BY_EDITION[edition])
