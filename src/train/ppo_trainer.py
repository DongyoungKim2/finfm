"""Thin wrapper around TRL's ``PPOTrainer``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from trl import PPOConfig, PPOTrainer

from src.model.timesfm_wrapper import TimesFMConfig, TimesFMForPPO
from .reward import reward_fn


@dataclass
class TrainPaths:
    data_path: str
    val_path: str


class TimesFMPPOTrainer:
    """Simple PPO training helper."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        model_cfg = TimesFMConfig(cfg.base_ckpt, cfg.lora_r)
        self.model = TimesFMForPPO(model_cfg)
        self.ppo_cfg = PPOConfig(**cfg.ppo)

        data = np.load(cfg.data_path)
        X, y = data["X"], data["y"]
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        self.loader = DataLoader(TensorDataset(tensor_x, tensor_y),
                                 batch_size=self.ppo_cfg.rollout_batch_size,
                                 shuffle=True)

    def train(self) -> None:
        trainer = PPOTrainer(self.ppo_cfg, self.model, ref_model=None, tokenizer=None)
        for _ in range(self.ppo_cfg.epochs):
            for X, y in self.loader:
                pred, _ = self.model(X)
                prev_close = X[:, -1, 0]
                rewards = reward_fn(pred.squeeze(-1), y.squeeze(-1), prev_close)
                trainer.step(X, X, rewards)

        Path("checkpoints").mkdir(exist_ok=True)
        self.model.timesfm.save_pretrained("checkpoints/timesfm")
