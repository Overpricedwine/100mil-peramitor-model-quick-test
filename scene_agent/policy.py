from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PolicyChoice:
    operation: int
    x: int
    y: int
    z: int
    scale: int
    color: int
    log_probability: torch.Tensor


class CudaFunctionPolicy(nn.Module):
    """A small trainable policy that emits one allowed function, not arbitrary code.

    It is intentionally modest: use it to prove the RL/render loop, then replace
    it with a fine-tuned text model adapter that emits the same schema.
    """

    OPERATIONS = ("add_block", "add_cylinder", "delete_function", "finish")

    def __init__(self, scene_feature_count: int = 12, hidden_size: int = 256, bins: int = 17) -> None:
        super().__init__()
        self.bins = bins
        self.byte_embedding = nn.Embedding(256, hidden_size)
        self.network = nn.Sequential(
            nn.Linear(hidden_size + scene_feature_count, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.operation_head = nn.Linear(hidden_size, len(self.OPERATIONS))
        self.axis_head = nn.Linear(hidden_size, bins * 3)
        self.scale_head = nn.Linear(hidden_size, bins)
        self.color_head = nn.Linear(hidden_size, 6)

    def forward(self, prompt_bytes: torch.Tensor, scene_features: torch.Tensor) -> dict[str, torch.Tensor]:
        # prompt_bytes: [batch, sequence] unsigned-byte token IDs
        prompt = self.byte_embedding(prompt_bytes.long()).mean(dim=1)
        hidden = self.network(torch.cat((prompt, scene_features.float()), dim=-1))
        return {
            "operation": self.operation_head(hidden),
            "axes": self.axis_head(hidden).view(-1, 3, self.bins),
            "scale": self.scale_head(hidden),
            "color": self.color_head(hidden),
        }

    def sample(self, prompt_bytes: torch.Tensor, scene_features: torch.Tensor) -> PolicyChoice:
        logits = self.forward(prompt_bytes, scene_features)
        distributions = {
            "operation": torch.distributions.Categorical(logits=logits["operation"]),
            "x": torch.distributions.Categorical(logits=logits["axes"][:, 0]),
            "y": torch.distributions.Categorical(logits=logits["axes"][:, 1]),
            "z": torch.distributions.Categorical(logits=logits["axes"][:, 2]),
            "scale": torch.distributions.Categorical(logits=logits["scale"]),
            "color": torch.distributions.Categorical(logits=logits["color"]),
        }
        values = {name: distribution.sample() for name, distribution in distributions.items()}
        log_probability = sum(distributions[name].log_prob(value) for name, value in values.items())
        return PolicyChoice(*(int(values[name][0]) for name in ("operation", "x", "y", "z", "scale", "color")), log_probability)
