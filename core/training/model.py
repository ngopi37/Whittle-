"""A small Llama-style decoder-only transformer.

The architecture (RMSNorm, rotary position embeddings, SwiGLU feed-forward, no
biases, untied embeddings) is chosen deliberately: it is exactly what GGUF's
``llama`` architecture expects, so a checkpoint trained here maps onto GGUF tensor
names with a rename, not a re-derivation — see ``core/runtime/gguf_export.py``. The
same checkpoint is what ``on-device-smoke-test`` (P5) actually loads with
``llama.cpp``, which is the real verification that the export is correct, not just
that this module runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from schemas.model import ModelConfig


@dataclass(frozen=True)
class LlamaConfig:
    """Concrete, buildable hyperparameters for :class:`LlamaModel`."""

    vocab_size: int
    hidden_size: int
    layers: int
    attention_heads: int
    intermediate_size: int
    context_length: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.attention_heads


def config_from_model(model_config: ModelConfig, *, vocab_size: int) -> LlamaConfig:
    """Derive a buildable :class:`LlamaConfig` from the project's ``ModelConfig``.

    ``vocab_size`` comes from the trained tokenizer's actual vocabulary (which may
    be smaller than ``model_config.vocabulary_size`` on a small corpus), not the
    catalog target, so the embedding table matches the tokenizer that will encode
    for it.
    """
    if model_config.hidden_size % model_config.attention_heads != 0:
        raise ValueError(
            f"hidden_size {model_config.hidden_size} is not divisible by "
            f"attention_heads {model_config.attention_heads}."
        )
    raw_intermediate = int(model_config.hidden_size * model_config.ffn_multiplier)
    intermediate_size = max(8, ((raw_intermediate + 7) // 8) * 8)
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=model_config.hidden_size,
        layers=model_config.layers,
        attention_heads=model_config.attention_heads,
        intermediate_size=intermediate_size,
        context_length=model_config.context_length,
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


def _rotary_cos_sin(
    seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    embedded = torch.cat([freqs, freqs], dim=-1)
    return embedded.cos().to(dtype), embedded.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.heads = cfg.attention_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(out)  # type: ignore[no-any-return]


class SwiGLU(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore[no-any-return]


class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    """Decoder-only transformer; tensor names match GGUF's ``llama`` layout 1:1."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.output_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.output = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        cos, sin = _rotary_cos_sin(
            seq_len,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            input_ids.device,
            self.token_embd.weight.dtype,
        )
        x = self.token_embd(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.output_norm(x)
        logits = self.output(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
