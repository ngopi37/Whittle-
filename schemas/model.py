"""Model architecture and training data contracts."""

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Initial architecture profile for a supported model size."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    target_parameters: int = Field(gt=0)
    vocabulary_size: int = Field(gt=0)
    context_length: int = Field(gt=0)
    layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    ffn_multiplier: float = Field(gt=0)


class TrainingConfig(BaseModel):
    """Training knobs that influence resource sizing and, later, the trainer."""

    mode: str = "pretraining"
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=0.0003, gt=0)
    epochs: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    mixed_precision: bool = False
