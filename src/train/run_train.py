"""Command line entry point for PPO training."""
from __future__ import annotations

import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra

from src.train.ppo_trainer import TrainerConfig, train_ppo


@hydra.main(config_path="../../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    ticker = cfg.get("ticker")
    gitsha = cfg.get("gitsha", "dev")
    trainer_cfg = TrainerConfig(
        base_ckpt=cfg.base_ckpt,
        lora_r=cfg.lora_r if cfg.lora_r > 0 else None,
        data_path=cfg.data_path,
        val_path=cfg.val_path,
        ppo=OmegaConf.to_container(cfg.ppo, resolve=True),
    )

    with mlflow.start_run(run_name=f"PPO_{ticker}_{gitsha}"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        model = train_ppo(trainer_cfg)
        mlflow.pytorch.log_model(model, "model")
        mlflow.set_tag("ticker", ticker)


if __name__ == "__main__":
    main()
