"""Reward utilities for PPO training."""
from __future__ import annotations

import torch


def reward_fn(pred_close: torch.Tensor, next_close: torch.Tensor, prev_close: torch.Tensor) -> torch.Tensor:
    """Return negative absolute return error scaled to [-2, 0]."""
    pred_ret = (pred_close - prev_close) / prev_close
    true_ret = (next_close - prev_close) / prev_close
    err = (pred_ret - true_ret).abs()
    reward = -err.clamp(0, 2)
    return reward
