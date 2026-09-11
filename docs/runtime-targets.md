# On-Device Runtime Targets

SG2 targets three on-device runtimes, and all three are **free-tier** in principle
— only distributing packages to a fleet is paid. **As implemented today, only
GGUF/llama.cpp is real.** ExecuTorch and TFLite/Core ML are deliberately deferred,
not silently missing — see "Why only GGUF so far" below.

| Target | Feature id | Primary platforms | Package artifact | Status |
|---|---|---|---|---|
| GGUF / llama.cpp | `runtime:gguf` | Windows, Linux, macOS, mobile forks | `.gguf` in a `.zip` + manifest | **implemented** |
| ExecuTorch | `runtime:executorch` | Android, iOS | `.pte` + manifest | deferred |
| TFLite / Core ML | `runtime:tflite-coreml` | Android (TFLite), Apple (Core ML) | `.tflite` / `.mlpackage` + manifest | deferred |

## GGUF / llama.cpp (implemented)

- **Architecture:** `core.training.model.LlamaModel` is deliberately a Llama-style
  decoder (RMSNorm, rotary position embeddings, SwiGLU feed-forward, no biases,
  untied embeddings) — the same tensor layout GGUF's `llama` architecture expects,
  so export is a rename, not a re-derivation.
- **Export:** `core.runtime.gguf_export.export_gguf` writes the checkpoint straight
  to GGUF via the `gguf` package's own `GGUFWriter` — the same library llama.cpp's
  `convert_hf_to_gguf.py` uses — embedding the byte-level BPE vocab/merges/special
  tokens directly in the file (llama.cpp needs no separate tokenizer file).
- **Quantization:** `fp16`, `q8_0` (default), `q4_0` — using `gguf.quants.quantize`,
  the reference implementation, not a hand-rolled approximation. Norm weights and
  the token embedding always stay `float32` regardless of quant type; mixing
  quantized/half-precision norms with the `float32` activations ggml's RMSNorm
  computes in causes a dtype mismatch at load time (found by hand, avoided by
  design). `Q5_K_M`/`Q4_K_M` (the "k-quant" superblock formats) are not
  implemented yet — `gguf.quants` supports them, so adding them is mostly wiring,
  not new quantization math; deferred for now, not because it's hard.
- **Package:** `package.zip` containing the `.gguf` (tokenizer already embedded)
  plus `manifest.json` (model size, quant type, tokenizer hash, source run id,
  sizing recommendation for the profiled device).
- **Smoke test:** `on-device-smoke-test` extracts the package and loads it with
  **real `llama.cpp`** via `llama-cpp-python` (`from llama_cpp import Llama`),
  generates a configurable prompt/token count, and records tokens/sec and process
  RSS — this is the actual proof the exported file works on the real runtime, not
  just that our own writer ran without raising. See
  `tests/unit/test_gguf_pipeline.py`.

## Why only GGUF so far

ExecuTorch and TFLite/Core ML each pull in a *different*, heavy toolchain
(`torch.export`/ExecuTorch's own runtime; `tensorflow` + `coremltools`), and Core
ML in particular can't be meaningfully verified without macOS — there's no way to
actually run the exported `.mlpackage` on this Windows-first project's own CI/dev
boxes the way `on-device-smoke-test` does for GGUF. Rather than ship export code
for those two that nothing here has ever actually executed, they stay
`DeferredStage`d (roadmap P4/P5, GGUF is done) until there's a way to verify them
for real, the same way GGUF is verified today.

### ExecuTorch (deferred)

- **Export:** lower the model through `torch.export` → ExecuTorch, XNNPACK backend
  by default; Core ML / Vulkan / QNN delegates as opt-in.
- **Quantization:** post-training dynamic/static INT8 via the ExecuTorch quantizer.
- **Package:** `.pte` + tokenizer + manifest (target backend, op coverage report).
- **Smoke test:** run the `.pte` with the ExecuTorch runtime on host (and, when a
  device/emulator is configured, on-device).

### TFLite / Core ML (deferred)

- **TFLite export:** checkpoint → SavedModel/ONNX → TFLite converter; INT8 with a
  small representative dataset drawn from the val split.
- **Core ML export:** via `coremltools`; INT8 / palettized weights; Apple Neural
  Engine where available.
- **Package:** `.tflite` (NNAPI-friendly) or `.mlpackage`, tokenizer, manifest.
- **Smoke test:** TFLite interpreter on host; Core ML on macOS runners.

## Manifest

Every package carries a `manifest.json`: model size, quantization, tokenizer hash,
runtime + version, target platform(s), the source run id, and the sizing
recommendation for the profiled device. The registry (P6) indexes packages by this
manifest.
