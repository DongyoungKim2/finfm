from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.serving.predictor import Predictor
from .loader import get_walk_forward_slices
from .strategy import BaseStrategy
from .engine import BacktestEngine
from .metrics import compute_all_metrics
from .report import log_to_mlflow


@hydra.main(config_path="../../configs", config_name="backtest.yaml", version_base=None)
def run(cfg: DictConfig) -> None:
    predictor = Predictor(cfg.mlflow_uri, cfg.ticker)
    slices = get_walk_forward_slices(
        predictor.meta,
        cfg.start,
        cfg.end,
        cfg.train_days,
        cfg.test_days,
    )
    strat: BaseStrategy = hydra.utils.instantiate(cfg.strategy)
    engine = BacktestEngine(predictor, strat)
    df_res = engine.walk_forward(slices)
    metrics = compute_all_metrics(df_res)
    log_to_mlflow(metrics, df_res, cfg)


if __name__ == "__main__":
    run()
