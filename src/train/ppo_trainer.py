"""PPO training loop using Hugging Face TRL."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig

from src.model.timesfm_wrapper import load_model
from src.model.reward import reward_fn


def train(config_path: str) -> None:
    """Train the TimesFM model with PPO."""
    # Placeholder example loading config
    config = PPOConfig()
    model, tokenizer = load_model()
    trainer = PPOTrainer(model, ref_model=None, tokenizer=tokenizer, config=config)

    # Example training loop using dummy data
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 16))
    dummy_label = torch.randn(1, 1)
    rewards = reward_fn(torch.randn(1, 1), dummy_label)
    trainer.step(dummy_input, dummy_input, rewards)

    Path("mlruns").mkdir(exist_ok=True)
    # Save model checkpoint (placeholder)
    trainer.save_pretrained("mlruns/latest")
