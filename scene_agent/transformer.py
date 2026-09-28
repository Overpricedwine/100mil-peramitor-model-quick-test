"""A ~100M causal Transformer for one-function-at-a-time scene generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    # 16,384 vocab + 2,048 positions + 12 x 768-wide blocks is ~99.2M params.
    vocab_size: int = 16_384
    context_length: int = 2_048
    hidden_size: int = 768
    layer_count: int = 12
    attention_heads: int = 12
    feed_forward_size: int = 3_072
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")


class RMSNorm(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return values * scale * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_count = config.attention_heads
        self.head_size = config.hidden_size // config.attention_heads
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = values.shape
        qkv = self.query_key_value(values)
        query, key, value = qkv.chunk(3, dim=-1)
        shape = (batch, tokens, self.head_count, self.head_size)
        query, key, value = (item.view(shape).transpose(1, 2) for item in (query, key, value))
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, hidden)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.contract = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.contract(F.gelu(self.expand(values), approximate="tanh")))


class TransformerBlock(nn.Module):
    """RMSNorm → causal attention → residual → RMSNorm → MLP → residual."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values))
        return values + self.feed_forward(self.feed_forward_norm(values))


class SceneFunctionTransformer(nn.Module):
    """Decoder-only Transformer which produces a single grammar-checked JSON function."""

    def __init__(self, config: TransformerConfig = TransformerConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layer_count))
        self.final_norm = RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying lowers memory use and keeps the default architecture near 100M parameters.
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialise)

    @staticmethod
    def _initialise(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, token_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        _, token_count = token_ids.shape
        if token_count > self.config.context_length:
            raise ValueError(f"input has {token_count} tokens; max is {self.config.context_length}")
        positions = torch.arange(token_count, device=token_ids.device)
        values = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            values = block(values)
        logits = self.output(self.final_norm(values))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                               labels[:, 1:].reshape(-1), ignore_index=-100)
        return result

    @torch.inference_mode()
    def generate(self, token_ids: torch.Tensor, *, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_k: int = 50, eos_token_id: int | None = None,
                 logits_processor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
        """Generate tokens, optionally masking choices before sampling."""
        for _ in range(max_new_tokens):
            window = token_ids[:, -self.config.context_length:]
            logits = self(window)["logits"][:, -1, :]
            if logits_processor is not None:
                logits = logits_processor(logits, token_ids)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    threshold = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return token_ids


def default_parameter_count() -> int:
    """Returns 99,192,576 for the default 12-layer configuration."""
    return SceneFunctionTransformer().parameter_count
