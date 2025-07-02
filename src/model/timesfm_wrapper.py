"""TimesFM wrapper with value and policy heads.

This module provides :class:`TimesFMForPPO`, a light wrapper around the
Hugging Face ``google/timesfm`` checkpoints.  The wrapper exposes a
``forward`` method that returns both the predicted price delta and a
value estimate so that it can be optimised with PPO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    def get_peft_model(model, config):  # type: ignore
        return model


@dataclass
class TimesFMConfig:
    """Configuration for :class:`TimesFMForPPO`."""

    base_ckpt: str = "google/timesfm-1.0-200m-pytorch"
    lora_r: Optional[int] = None


class TimesFMForPPO(nn.Module):
    """Tiny wrapper adding LoRA, value and policy heads."""

    def __init__(self, cfg: TimesFMConfig):
        super().__init__()
        self.timesfm = AutoModel.from_pretrained(cfg.base_ckpt)
        hidden = self.timesfm.config.hidden_size
        if cfg.lora_r and LoraConfig is not None:
            lora_cfg = LoraConfig(r=cfg.lora_r, target_modules="all")
            self.timesfm = get_peft_model(self.timesfm, lora_cfg)
        self.value_head = nn.Linear(hidden, 1)
        self.policy_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, **kwargs):
        """Return predicted delta and value estimate."""
        out = self.timesfm(x, **kwargs)
        h = out.last_hidden_state[:, -1]
        pred = self.policy_head(h)
        value = self.value_head(h)
        return pred, value
