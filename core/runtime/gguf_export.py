"""Export a trained checkpoint to GGUF, loadable by real ``llama.cpp``.

Tensor names and hyperparameter keys follow llama.cpp's ``llama`` architecture
exactly — that is the entire point of ``core.training.model.LlamaModel``'s design
— and quantization uses the reference implementation from the ``gguf`` package
(the same one llama.cpp's own ``convert_hf_to_gguf.py`` uses), not a hand-rolled
approximation. Correctness is verified by ``on-device-smoke-test`` actually loading
the file with ``llama-cpp-python`` and generating tokens, not just by this module
running without raising.

Norm weights and the token embedding always stay ``float32`` regardless of
``quant_type`` — mixing quantized/half-precision norm weights with the ``float32``
activations ggml's RMSNorm computes in causes a dtype-mismatch at load time
(verified by hand: only the projection/FFN matrices are safe to shrink).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

QUANT_TYPES: dict[str, gguf.GGMLQuantizationType | None] = {
    "fp16": None,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}


def export_gguf(
    *,
    checkpoint_path: Path,
    tokenizer_path: Path,
    out_path: Path,
    quant_type: str,
) -> dict[str, Any]:
    """Write ``out_path`` and return export stats (tensor count, byte size, quant_type)."""
    if quant_type not in QUANT_TYPES:
        raise ValueError(
            f"Unknown GGUF quant type {quant_type!r}; expected one of {sorted(QUANT_TYPES)}."
        )
    qtype = QUANT_TYPES[quant_type]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state"]
    layers = int(config["layers"])

    writer = gguf.GGUFWriter(str(out_path), "llama")
    _write_hyperparameters(writer, config)
    _write_tokenizer(writer, tokenizer_path)

    tensor_count = 0
    for gguf_name, torch_key, quantizable in _tensor_plan(layers):
        tensor_count += _add_weight(
            writer, gguf_name, state_dict[torch_key], qtype, quantizable=quantizable
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "quant_type": quant_type,
        "tensors": tensor_count,
        "bytes": out_path.stat().st_size,
        "vocab_size": int(config["vocab_size"]),
        "layers": layers,
    }


def _tensor_plan(layers: int) -> list[tuple[str, str, bool]]:
    """(gguf tensor name, checkpoint state_dict key, quantizable) for every tensor.

    ``quantizable=False`` marks norms and the token embedding, which must stay
    ``float32`` regardless of ``quant_type`` (see module docstring).
    """
    plan: list[tuple[str, str, bool]] = [("token_embd.weight", "token_embd.weight", False)]
    per_layer = [
        ("attn_norm.weight", "attn_norm.weight", False),
        ("attn_q.weight", "attn.q_proj.weight", True),
        ("attn_k.weight", "attn.k_proj.weight", True),
        ("attn_v.weight", "attn.v_proj.weight", True),
        ("attn_output.weight", "attn.o_proj.weight", True),
        ("ffn_norm.weight", "ffn_norm.weight", False),
        ("ffn_gate.weight", "mlp.gate_proj.weight", True),
        ("ffn_up.weight", "mlp.up_proj.weight", True),
        ("ffn_down.weight", "mlp.down_proj.weight", True),
    ]
    for i in range(layers):
        plan.extend(
            (f"blk.{i}.{gguf_suffix}", f"blocks.{i}.{torch_suffix}", quantizable)
            for gguf_suffix, torch_suffix, quantizable in per_layer
        )
    plan.append(("output_norm.weight", "output_norm.weight", False))
    plan.append(("output.weight", "output.weight", True))
    return plan


def _write_hyperparameters(writer: gguf.GGUFWriter, config: dict[str, Any]) -> None:
    writer.add_name("sg2-on-device")
    writer.add_vocab_size(int(config["vocab_size"]))
    writer.add_context_length(int(config["context_length"]))
    writer.add_embedding_length(int(config["hidden_size"]))
    writer.add_block_count(int(config["layers"]))
    writer.add_feed_forward_length(int(config["intermediate_size"]))
    writer.add_head_count(int(config["attention_heads"]))
    writer.add_head_count_kv(int(config["attention_heads"]))  # no GQA: kv heads == attn heads
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_freq_base(float(config["rope_theta"]))


def _write_tokenizer(writer: gguf.GGUFWriter, tokenizer_path: Path) -> None:
    """Embed the byte-level BPE vocab/merges as llama.cpp's ``gpt2`` tokenizer expects."""
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = payload["model"]
    vocab: dict[str, int] = model["vocab"]
    merges: list[Any] = model.get("merges", [])

    id_to_token: list[str] = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        id_to_token[token_id] = token

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(id_to_token)
    writer.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in merges])
    writer.add_token_types([gguf.TokenType.NORMAL] * len(id_to_token))

    for setter, name in (
        (writer.add_bos_token_id, "<bos>"),
        (writer.add_eos_token_id, "<eos>"),
        (writer.add_unk_token_id, "<unk>"),
        (writer.add_pad_token_id, "<pad>"),
    ):
        special_id = vocab.get(name)
        if special_id is not None:
            setter(special_id)


def _add_weight(
    writer: gguf.GGUFWriter,
    name: str,
    tensor: torch.Tensor,
    qtype: gguf.GGMLQuantizationType | None,
    *,
    quantizable: bool = True,
) -> int:
    """Write one tensor, quantized per ``qtype`` unless it's a norm/embedding (see docstring)."""
    array = tensor.detach().numpy().astype(np.float32)
    if not quantizable:
        writer.add_tensor(name, array)
    elif qtype is None:
        writer.add_tensor(name, array.astype(np.float16))
    else:
        from gguf.quants import quantize

        writer.add_tensor(name, quantize(array, qtype), raw_dtype=qtype)
    return 1
