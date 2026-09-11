# Enterprise Roadmap

The free edition keeps all model-building logic local. Enterprise adapters add
governance and scale **behind interfaces**, without changing the core engine or the
stage contracts.

Planned provider interfaces (roadmap P6–P7, see
[product/roadmap.md](product/roadmap.md)):

- **Identity** — SSO / SCIM; maps users to shared projects and roles.
- **Policy** — allowed data sources, model sizes, and runtime targets; evaluated
  before a `pipeline run`.
- **Audit** — append-only sink for run events, artifact hashes, and policy decisions.
- **Model registry** — versioned package storage indexed by the package manifest
  (`docs/runtime-targets.md`).
- **Deployment** — staged fleet rollout with rollback.

Each is a `core`-level interface with a local no-op default and a paid
implementation. The feature/edition boundary is in
[product/editions.md](product/editions.md); the gating mechanism is
`core/editions.py`.
