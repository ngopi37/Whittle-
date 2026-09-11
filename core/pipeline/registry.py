"""The canonical on-device model build pipeline."""

from __future__ import annotations

from core.editions import (
    FEATURE_FLEET_DEPLOY,
    FEATURE_MODEL_REGISTRY,
    FEATURE_REMOTE_TRAINING,
)
from core.pipeline.base import Stage
from core.pipeline.stages.data_ingest import DataIngestStage
from core.pipeline.stages.evaluate import EvaluateStage
from core.pipeline.stages.model_init import ModelInitStage
from core.pipeline.stages.on_device_smoke_test import OnDeviceSmokeTestStage
from core.pipeline.stages.package import PackageStage
from core.pipeline.stages.pretrain import PretrainStage
from core.pipeline.stages.quantize import QuantizeStage
from core.pipeline.stages.tokenizer_train import TokenizerTrainStage
from schemas.pipeline import STAGE_ORDER, StageName


def build_default_pipeline() -> list[Stage]:
    """Return the ordered stages of the standard local build.

    Every canonical stage is implemented. ``quantize``/``package``/
    ``on-device-smoke-test`` currently support the GGUF/llama.cpp runtime target
    only — ExecuTorch and TFLite/Core ML are documented in
    docs/runtime-targets.md as deliberately deferred, not silently missing. Local
    training/eval/quantize/package are free-tier; only remote/registry/deploy
    concerns require a paid edition.
    """
    stages: list[Stage] = [
        DataIngestStage(),
        TokenizerTrainStage(),
        ModelInitStage(),
        PretrainStage(),
        EvaluateStage(),
        QuantizeStage(),
        PackageStage(),
        OnDeviceSmokeTestStage(),
    ]
    _assert_canonical_order(stages)
    return stages


#: Features required to run a stage remotely / at fleet scale (paid). The local
#: pipeline itself is free; these gate the scale-out variants.
REMOTE_STAGE_FEATURES: dict[StageName, str] = {
    "pretrain": FEATURE_REMOTE_TRAINING,
    "package": FEATURE_MODEL_REGISTRY,
    "on-device-smoke-test": FEATURE_FLEET_DEPLOY,
}


def _assert_canonical_order(stages: list[Stage]) -> None:
    names = tuple(stage.name for stage in stages)
    if names != STAGE_ORDER:
        raise RuntimeError(f"Pipeline order {names} does not match canonical {STAGE_ORDER}.")
