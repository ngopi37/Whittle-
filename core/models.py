"""Initial model architecture profile generation."""

from pathlib import Path
from typing import Any

import yaml

from core.catalog import USER_FACING_SIZES, get_spec
from schemas.model import ModelConfig

__all__ = ["PHASE_1_PROFILE_NAMES", "PROFILE_DIR", "ModelConfig", "ModelConfigGenerator"]

PROFILE_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"

#: Sizes offered by the Phase 1 user-facing flows. Derived from the catalog so the
#: recommendation surface and the generator never drift apart.
PHASE_1_PROFILE_NAMES = USER_FACING_SIZES


class ModelConfigGenerator:
    """Generate an initial architecture profile for a supported model size."""

    def __init__(self, profile_dir: Path = PROFILE_DIR) -> None:
        self.profile_dir = profile_dir

    def generate(
        self,
        model_size: str,
        vocabulary_size: int | None = None,
        context_length: int | None = None,
        layers: int | None = None,
        hidden_size: int | None = None,
        attention_heads: int | None = None,
        ffn_multiplier: float | None = None,
    ) -> ModelConfig:
        """Return a profile with optional architecture overrides.

        Raises ``ValueError`` for sizes outside the catalog. Edition gating for larger
        sizes is enforced by callers (see :mod:`core.editions`), not here.
        """
        spec = get_spec(model_size)
        raw = self._load_profile(spec.profile_name)
        overrides = {
            "vocabulary_size": vocabulary_size,
            "context_length": context_length,
            "layers": layers,
            "hidden_size": hidden_size,
            "attention_heads": attention_heads,
            "ffn_multiplier": ffn_multiplier,
        }
        raw.update({key: value for key, value in overrides.items() if value is not None})
        return ModelConfig.model_validate(raw)

    def _load_profile(self, profile_name: str) -> dict[str, Any]:
        """Load one YAML model profile from the configured profile directory."""
        path = self.profile_dir / f"{profile_name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model profile: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Invalid model profile (expected a mapping): {path}")
        return raw
