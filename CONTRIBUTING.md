# Contributing

- Python 3.12+.
- Keep `core` logic independent of UI frameworks; `apps/` is presentation only.
- Cross-module contracts are Pydantic models in `schemas/`.
- New pipeline stages: implement `Stage`, swap the `DeferredStage` in
  `core/pipeline/registry.py`, keep `plan()` side-effect free. See
  [docs/pipeline.md](docs/pipeline.md).
- Local capabilities stay in the free edition; gate scale/team/governance features
  via `core/editions.py`. See [docs/product/editions.md](docs/product/editions.md).
- Add focused tests for public behavior.
- Before submitting: `pytest`, `ruff check .`, and `mypy .` must all pass.
