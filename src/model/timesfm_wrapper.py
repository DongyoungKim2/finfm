"""TimesFM model wrapper with optional LoRA adapters and value head."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel


__all__ = ["TimesFMForPPO"]


def _maybe_add_lora(model: nn.Module, r: Optional[int]) -> nn.Module:
    """Add LoRA adapters to *model* if ``r`` is specified."""
    if not r:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception:  # peft may not be installed
        return model

    config = LoraConfig(r=r, lora_alpha=r, target_modules=None)
    return get_peft_model(model, config)


class TimesFMForPPO(nn.Module):
    """Tiny wrapper around ``google/timesfm`` for PPO fine-tuning."""

    def __init__(self, base_ckpt: str, lora_r: Optional[int] = None, num_stocks: int | None = None, emb_dim: int = 16) -> None:
        super().__init__()
        self.timesfm = AutoModel.from_pretrained(base_ckpt)
        hidden_size = self.timesfm.config.hidden_size
        self.timesfm = _maybe_add_lora(self.timesfm, lora_r)

        if num_stocks:
            self.id_emb = nn.Embedding(num_stocks, emb_dim)
            hidden_size += emb_dim
        else:
            self.id_emb = None

        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, stock_id: Optional[torch.LongTensor] = None):
        """Return price delta prediction and value estimate."""
        if self.id_emb is not None and stock_id is not None:
            emb = self.id_emb(stock_id).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb], dim=-1)

        features = self.timesfm(inputs_embeds=x).last_hidden_state
        value = self.value_head(features)
        pred = self.policy_head(features)
        return pred, value
