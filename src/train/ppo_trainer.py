"""Lightweight PPO training orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from trl import PPOConfig, PPOTrainer

from src.model.timesfm_wrapper import TimesFMForPPO
from src.train.reward import reward_fn


@dataclass
class TrainerConfig:
    base_ckpt: str
    lora_r: int | None
    data_path: str
    val_path: str
    ppo: dict


def _load_dataset(path: str) -> TensorDataset:
    data = np.load(path)
    x = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    return TensorDataset(x, y)


def train_ppo(cfg: TrainerConfig) -> TimesFMForPPO:
    """Train a TimesFM model with PPO using *cfg*."""
    model = TimesFMForPPO(cfg.base_ckpt, cfg.lora_r)
    ppo_cfg = PPOConfig(**cfg.ppo)

    ds = _load_dataset(cfg.data_path)
    loader = DataLoader(ds, batch_size=ppo_cfg.rollout_batch_size, shuffle=True)

    trainer = PPOTrainer(ppo_cfg, model, ref_model=None, tokenizer=None)

    for _ in range(ppo_cfg.epochs):
        for batch in loader:
            queries = batch[0]
            true_ret = batch[1]
            pred, values = model(queries)
            rewards = reward_fn(pred.squeeze(-1), true_ret.squeeze(-1))
            trainer.step(queries, pred, rewards)

    return model
