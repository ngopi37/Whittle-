"""Offline CLI for hardware intelligence and the local model-build pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from core.catalog import ALL_SIZES, get_spec
from core.editions import FeatureNotAvailable, resolve_entitlements
from core.hardware.profiler import HardwareProfiler
from core.pipeline.base import StageError, StageNotImplemented
from core.pipeline.pipeline import Pipeline
from core.project import ProjectConfig, create_project, refresh_recommendation
from core.sizing.engine import FitResult, ModelFitEngine
from schemas.model import TrainingConfig

_FIT_LIST = TypeAdapter(list[FitResult])


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(prog="sg2-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("hardware", help="show detected hardware")
    subparsers.add_parser("edition", help="show the active edition and its features")

    recommend = subparsers.add_parser("recommend", help="recommend model sizes")
    _add_training_flags(recommend)

    project = subparsers.add_parser("project", help="manage local projects")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    create = project_sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--model", choices=list(ALL_SIZES), default="20M")
    create.add_argument("--output", type=Path, default=Path("project.json"))
    create.add_argument(
        "--with-hardware", action="store_true", help="embed current hardware recommendation"
    )
    show = project_sub.add_parser("show")
    show.add_argument("path", type=Path)
    refresh = project_sub.add_parser("refresh", help="re-profile and recompute the recommendation")
    refresh.add_argument("path", type=Path)

    pipeline = subparsers.add_parser("pipeline", help="plan or run the model-build pipeline")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command", required=True)
    plan = pipeline_sub.add_parser("plan")
    plan.add_argument("path", type=Path)
    run = pipeline_sub.add_parser("run")
    run.add_argument("path", type=Path)
    run.add_argument(
        "--stage", action="append", dest="stages", help="run only this stage (repeatable)"
    )
    run.add_argument(
        "--input", action="append", dest="inputs", default=[], help="data-ingest source file"
    )
    run.add_argument(
        "--algorithm",
        choices=["bpe", "unigram"],
        default="bpe",
        help="tokenizer-train algorithm",
    )
    run.add_argument(
        "--vocab-size", type=int, default=None, help="tokenizer-train vocabulary size override"
    )
    run.add_argument(
        "--no-dedup", action="store_false", dest="dedup", default=True,
        help="disable data-ingest exact-duplicate removal",
    )
    run.add_argument(
        "--val-ratio", type=float, default=0.0, help="data-ingest held-out validation fraction"
    )
    run.add_argument("--max-steps", type=int, default=None, help="pretrain step count")
    run.add_argument("--block-size", type=int, default=None, help="pretrain sequence length")
    run.add_argument(
        "--checkpoint-every", type=int, default=None, help="pretrain steps between checkpoints"
    )
    run.add_argument("--seed", type=int, default=None, help="pretrain RNG seed")
    run.add_argument(
        "--quant-type",
        choices=["fp16", "q8_0", "q4_0"],
        default=None,
        help="quantize GGUF quant type (default q8_0)",
    )
    run.add_argument(
        "--prompt", default=None, help="on-device-smoke-test generation prompt"
    )
    run.add_argument(
        "--max-tokens", type=int, default=None, help="on-device-smoke-test tokens to generate"
    )
    run.add_argument(
        "--n-ctx", type=int, default=None, help="on-device-smoke-test context window"
    )
    run.add_argument("--remote", action="store_true", help="request the scale-out (paid) variant")
    return parser


def _add_training_flags(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--batch-size", type=int, default=1)
    sub.add_argument("--context-length", type=int, default=None)
    sub.add_argument("--mixed-precision", action="store_true")


def _emit_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _training_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        batch_size=args.batch_size, mixed_precision=bool(args.mixed_precision)
    )


def _cmd_hardware() -> int:
    print(HardwareProfiler().profile().model_dump_json(indent=2))
    return 0


def _cmd_edition() -> int:
    ent = resolve_entitlements()
    _emit_json({"edition": ent.edition.value, "features": sorted(ent.features)})
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    profile = HardwareProfiler().profile()
    training = _training_from_args(args)
    engine = ModelFitEngine()
    if args.context_length is None:
        results = engine.recommend(profile, training)
    else:
        results = [
            engine.evaluate(
                profile,
                size,
                training=training,
                model_config=_with_context(size, args.context_length),
            )
            for size in ("20M", "50M", "100M")
        ]
    print(_FIT_LIST.dump_json(results, indent=2).decode())
    return 0


def _with_context(size: str, context_length: int):  # type: ignore[no-untyped-def]
    from core.models import ModelConfigGenerator

    return ModelConfigGenerator().generate(size, context_length=context_length)


def _cmd_project_create(args: argparse.Namespace) -> int:
    _require_model_edition(args.model)
    hardware = HardwareProfiler().profile() if args.with_hardware else None
    recommendation = (
        ModelFitEngine().evaluate(hardware, args.model) if hardware is not None else None
    )
    project = create_project(
        args.name, args.model, hardware=hardware, recommendation=recommendation
    )
    project.save(args.output)
    print(f"Created {args.output}")
    return 0


def _cmd_project_show(args: argparse.Namespace) -> int:
    project = _load_project(args.path)
    print(project.model_dump_json(indent=2))
    return 0


def _cmd_project_refresh(args: argparse.Namespace) -> int:
    project = _load_project(args.path)
    updated = refresh_recommendation(project, HardwareProfiler().profile())
    updated.save(args.path)
    print(f"Refreshed {args.path}")
    return 0


def _cmd_pipeline_plan(args: argparse.Namespace) -> int:
    project = _load_project(args.path)
    plan = Pipeline().plan(project)
    print(plan.model_dump_json(indent=2))
    return 0


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    project = _load_project(args.path)
    try:
        handle = Pipeline().run(
            project,
            stages=args.stages,
            params={
                "inputs": args.inputs,
                "algorithm": args.algorithm,
                "vocab_size": args.vocab_size,
                "dedup": args.dedup,
                "val_ratio": args.val_ratio,
                "max_steps": args.max_steps,
                "block_size": args.block_size,
                "checkpoint_every": args.checkpoint_every,
                "seed": args.seed,
                "quant_type": args.quant_type,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "n_ctx": args.n_ctx,
            },
            remote=bool(args.remote),
        )
    except StageNotImplemented as exc:
        return _fail(str(exc))
    except StageError as exc:
        return _fail(str(exc))
    except KeyError as exc:
        return _fail(str(exc).strip('"'))
    print(f"Run {handle.record.id}: {handle.record.status}")
    print(f"  directory: {handle.dir}")
    for artifact in handle.record.artifacts:
        print(f"  artifact: {artifact.kind.value} -> {artifact.path}")
    return 0 if handle.record.status == "succeeded" else 1


def _require_model_edition(model_size: str) -> None:
    spec = get_spec(model_size)
    resolve_entitlements().require(spec.required_feature)


def _load_project(path: Path) -> ProjectConfig:
    try:
        return ProjectConfig.load(path)
    except FileNotFoundError:
        raise SystemExit(_fail(f"project file not found: {path}")) from None
    except ValueError as exc:
        raise SystemExit(_fail(f"invalid project file {path}: {exc}")) from None


def main(argv: list[str] | None = None) -> int:
    """Run a CLI command and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "hardware":
            return _cmd_hardware()
        if args.command == "edition":
            return _cmd_edition()
        if args.command == "recommend":
            return _cmd_recommend(args)
        if args.command == "project":
            if args.project_command == "create":
                return _cmd_project_create(args)
            if args.project_command == "show":
                return _cmd_project_show(args)
            if args.project_command == "refresh":
                return _cmd_project_refresh(args)
        if args.command == "pipeline":
            if args.pipeline_command == "plan":
                return _cmd_pipeline_plan(args)
            if args.pipeline_command == "run":
                return _cmd_pipeline_run(args)
    except FeatureNotAvailable as exc:
        return _fail(str(exc))
    return _fail(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
