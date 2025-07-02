"""Reward function for PPO fine-tuning."""
from __future__ import annotations

import torch


def mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute percentage error."""
    return (pred - target).abs() / target.abs().clamp(min=1e-5)


def reward_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple negative MAPE reward."""
    return -mape_loss(pred, target).mean()
