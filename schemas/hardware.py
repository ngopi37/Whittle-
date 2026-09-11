"""Normalized hardware data contracts."""

from pydantic import BaseModel, Field


class CpuProfile(BaseModel):
    vendor: str = "Unknown"
    model: str = "Unknown"
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)
    frequency_mhz: float | None = Field(default=None, ge=0)
    instruction_sets: list[str] = Field(default_factory=list)


class MemoryProfile(BaseModel):
    total_gb: float = Field(ge=0)
    available_gb: float = Field(ge=0)


class GpuProfile(BaseModel):
    vendor: str = "None"
    model: str = "None"
    memory_gb: float = Field(default=0, ge=0)
    available_memory_gb: float | None = Field(default=None, ge=0)


class StorageProfile(BaseModel):
    capacity_gb: float = Field(ge=0)
    free_gb: float = Field(ge=0)


class HardwareProfile(BaseModel):
    os: str
    architecture: str
    cpu: CpuProfile
    memory: MemoryProfile
    gpu: GpuProfile
    storage: StorageProfile

    @property
    def has_gpu(self) -> bool:
        """Return whether a locally detected GPU has usable memory metadata."""
        return self.gpu.vendor != "None" and self.gpu.memory_gb > 0
