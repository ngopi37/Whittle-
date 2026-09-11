# Editions

**Local is free, forever. You pay for scale, team, and governance.**

Everything that runs on a single machine — profiling, data prep, tokenizer training,
model training, evaluation, quantization, packaging, and on-device testing, for every
supported runtime target — is in the free edition and always will be. Paid editions
add the things that only matter once more than one machine or more than one person is
involved: distributed training, a shared model registry, fleet deployment, team
collaboration, and enterprise governance.

## Feature matrix

| Capability | Free | Pro | Enterprise |
|---|:--:|:--:|:--:|
| Hardware profiling & sizing | ✅ | ✅ | ✅ |
| Local pipeline: data → tokenizer → train → eval → quantize → package → smoke test | ✅ | ✅ | ✅ |
| Runtime targets: GGUF/llama.cpp, ExecuTorch, TFLite/Core ML | ✅ | ✅ | ✅ |
| Model sizes ≤ 100M | ✅ | ✅ | ✅ |
| Model sizes > 100M (200M, 500M, …) | — | ✅ | ✅ |
| Remote / multi-GPU training | — | ✅ | ✅ |
| Model registry (versioned packages) | — | ✅ | ✅ |
| Fleet deployment (staged rollout, rollback) | — | ✅ | ✅ |
| Teams & shared projects (RBAC) | — | ✅ | ✅ |
| SSO / SCIM | — | — | ✅ |
| Policy engine (data sources, sizes, targets) | — | — | ✅ |
| Audit sink (append-only) | — | — | ✅ |
| On-prem control plane | — | — | ✅ |

Feature identifiers live in `core/editions.py`; each is a stable string that also
appears in stage plans and license files.

## How an edition is resolved

Resolution is entirely offline — no network call, ever:

1. `SG2_EDITION` environment variable (`free` | `pro` | `enterprise`), else
2. `~/.sg2/license.json` — a local file: `{"edition": "pro", ...}`, else
3. `free`.

> Signature verification of `license.json` is not implemented yet (a `TODO` in
> `resolve_entitlements`). Until it is, the file is honored as-is; treat paid
> editions as trust-based for local development.

## FAQ

**Will a free feature ever move behind a paywall?** No. The local pipeline is the
product's foundation and stays free. New *scale/team/governance* capabilities may be
paid; new *local* capabilities will not.

**Can I self-host the paid control plane?** That is the intent for Enterprise
(`on-prem-control-plane`). See [../enterprise-roadmap.md](../enterprise-roadmap.md).

**What happens to my projects if a license expires?** Projects and run artifacts are
plain local files and remain fully usable on the free edition, minus the paid
capabilities (e.g. you keep local training of ≤100M models).
