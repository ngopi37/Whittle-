from pathlib import Path

import pytest

from core.models import ModelConfigGenerator


def test_model_profiles_load_from_yaml() -> None:
    config = ModelConfigGenerator().generate("50M")
    assert config.name == "50M"
    assert config.target_parameters == 50_000_000
    assert config.context_length == 1024


def test_model_profile_overrides_are_validated() -> None:
    config = ModelConfigGenerator().generate(
        "20M",
        vocabulary_size=20_000,
        context_length=768,
        layers=7,
        hidden_size=320,
        attention_heads=10,
        ffn_multiplier=3.5,
    )
    assert config.vocabulary_size == 20_000
    assert config.layers == 7
    assert config.attention_heads == 10


def test_catalog_covers_large_sizes() -> None:
    config = ModelConfigGenerator().generate("200M")
    assert config.target_parameters == 200_000_000


def test_unknown_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model size"):
        ModelConfigGenerator().generate("999M")


def test_missing_model_profile_fails_clearly(tmp_path: Path) -> None:
    generator = ModelConfigGenerator(profile_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match=r"20m\.yaml"):
        generator.generate("20M")
