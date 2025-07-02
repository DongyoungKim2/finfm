"""Reward utilities for PPO training."""
from __future__ import annotations

import torch


def abs_return_error(pred_ret: torch.Tensor, true_ret: torch.Tensor) -> torch.Tensor:
    """Absolute return error."""
    return (pred_ret - true_ret).abs()


def reward_fn(pred_ret: torch.Tensor, true_ret: torch.Tensor) -> torch.Tensor:
    """Scale negative absolute return error to ``[-2, 0]`` for PPO."""
    error = abs_return_error(pred_ret, true_ret)
    reward = -error
    return reward.clamp(min=-2.0, max=0.0)
