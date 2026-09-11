# Hardware Sizing

The fit engine (`core/sizing/engine.py`) turns a measured hardware profile plus the
project's `ModelConfig` and `TrainingConfig` into a conservative training-resource
estimate. Every figure is an explicit, named function of its inputs. Duration is a
range, not a promise.

## RAM components

| Component | Model |
|---|---|
| Model weights | `params × 4 B` (fp32) |
| Master copy | `params × 2 B` when mixed precision, else 0 |
| Gradients | `params × (2 B mixed / 4 B full)` |
| Optimizer | `params × 8 B` (Adam moments m, v in fp32) |
| Activations | `(feature + attention)` acts, ∝ `batch × context`, halved under mixed precision; scales with `hidden × layers`, `heads × context²`, and `ffn_multiplier` |
| Tokenizer | `0.05 GB + vocab × 8e-7` |
| Data pipeline | `0.2 GB + batch × 0.01` (shuffle/prefetch buffers) |
| Runtime overhead | `0.5 GB` + `1.0 GB` if a GPU is present (framework, dataloader workers, accelerator allocator) |

`component_subtotal` is the sum of the above. The **safety margin** is a transparent
`SAFETY_FRACTION` (0.25) of that subtotal — no hidden second heuristic. Peak RAM =
`component_subtotal + safety_margin`.

Dataset and checkpoint sizes are **disk**, not RAM, and are excluded from the RAM
total (this was previously conflated).

## Disk gating

`disk_required = checkpoint_storage + dataset_storage`. If free disk on the workspace
drive is below that, the status is capped at `constrained`; below half of it, the
status becomes `not_recommended`. `disk_required_gb` / `disk_available_gb` are on
every `FitResult`.

## Status and confidence

From the `available / required` RAM ratio: `≥ 2.0` recommended, `≥ 1.25` possible,
`≥ 0.9` constrained, else not recommended. CPU-only recommendations are capped at
`medium` confidence.

## Model catalog

`core/catalog.py` defines 20M / 50M / 100M (free) and 200M / 500M (paid). User-facing
`recommend` reports the three free sizes; `evaluate` accepts any catalog size (paid
sizes are gated at the CLI, not in the engine).

## GPU memory accuracy

`Win32_VideoController.AdapterRAM` is a 32-bit field and saturates near 4 GB. The
profiler also reads `HardwareInformation.qwMemorySize` from the display-class
registry key and takes the larger value, so cards above 4 GB report correctly when
the driver publishes it. If neither source is reliable, treat GPU memory as a lower
bound.

## Example

A 16 GB Intel i7-10750H, CPU-only, 12 GB available, default `TrainingConfig`:
20M / 50M / 100M all classify `recommended` with rising RAM estimates. Raising
`--batch-size` to 16 pushes 50M and 100M past the available-memory limit — the knob
now visibly changes the answer.
