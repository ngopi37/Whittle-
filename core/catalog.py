"""Single source of truth for supported model sizes.

Every other module derives its notion of "which sizes exist" from :data:`MODELS`
instead of hard-coding parallel lists.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.editions import FEATURE_MODEL_SIZE_LARGE, FEATURE_MODEL_SIZE_SMALL


@dataclass(frozen=True)
class ModelSpec:
    """Catalog entry for one model size."""

    key: str
    parameters: int
    profile_name: str
    required_feature: str
    user_facing: bool


MODELS: dict[str, ModelSpec] = {
    "20M": ModelSpec("20M", 20_000_000, "20m", FEATURE_MODEL_SIZE_SMALL, user_facing=True),
    "50M": ModelSpec("50M", 50_000_000, "50m", FEATURE_MODEL_SIZE_SMALL, user_facing=True),
    "100M": ModelSpec("100M", 100_000_000, "100m", FEATURE_MODEL_SIZE_SMALL, user_facing=True),
    "200M": ModelSpec("200M", 200_000_000, "200m", FEATURE_MODEL_SIZE_LARGE, user_facing=False),
    "500M": ModelSpec("500M", 500_000_000, "500m", FEATURE_MODEL_SIZE_LARGE, user_facing=False),
}

#: Model sizes surfaced in user-facing recommendation flows, in ascending order.
USER_FACING_SIZES: tuple[str, ...] = tuple(
    key for key, spec in MODELS.items() if spec.user_facing
)

#: Every catalog key, in ascending order.
ALL_SIZES: tuple[str, ...] = tuple(MODELS)

#: Parameter counts keyed by size, for callers that only need the number.
MODEL_PARAMETERS: dict[str, int] = {key: spec.parameters for key, spec in MODELS.items()}


def get_spec(model_size: str) -> ModelSpec:
    """Return the catalog entry for a size or raise ``ValueError``."""
    try:
        return MODELS[model_size]
    except KeyError:
        raise ValueError(
            f"Unsupported model size: {model_size!r}. Known sizes: {', '.join(MODELS)}"
        ) from None
