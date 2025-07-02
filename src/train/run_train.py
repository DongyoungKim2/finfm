"""CLI entry-point to train TimesFM with PPO."""
from __future__ import annotations

from pathlib import Path

import mlflow
from hydra import compose, initialize
from omegaconf import OmegaConf

from .ppo_trainer import TimesFMPPOTrainer


def main(config: str = "configs/train.yaml", **overrides) -> None:
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name=Path(config).name, overrides=overrides)
    mlflow.autolog()
    with mlflow.start_run(run_name=f"PPO_{cfg.get('ticker', 'UNK')}"):
        mlflow.log_params(OmegaConf.to_container(cfg))
        trainer = TimesFMPPOTrainer(cfg)
        trainer.train()
        mlflow.pytorch.log_model(trainer.model.timesfm, "model")


if __name__ == "__main__":
    main()
