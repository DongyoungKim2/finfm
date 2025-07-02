"""Command line interface for the data pipeline."""
from __future__ import annotations

import os
from pathlib import Path
import joblib

import mlflow
import numpy as np
import typer
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.data import (
    fetch_prices,
    repair_calendar,
    engineer_features,
    make_windows,
    Scaler,
    walk_forward_split,
    holdout_split,
)

cli = typer.Typer()


@cli.command()
def download(config: str = "configs/data.yaml"):
    """Fetch raw prices and cache them."""
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name=Path(config).name)
    with mlflow.start_run(run_name="data_prep"):
        mlflow.log_params(OmegaConf.to_container(cfg))
        fetch_prices(
            symbols=cfg.symbols,
            start=cfg.start,
            end=cfg.end,
            vendor=cfg.vendor,
            interval=cfg.interval,
        )
        mlflow.log_artifacts("data/raw")


@cli.command()
def preprocess(config: str = "configs/data.yaml"):
    """Clean, engineer features and scale."""
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name=Path(config).name)
    with mlflow.start_run(run_name="data_prep"):
        mlflow.log_params(OmegaConf.to_container(cfg))
        df = fetch_prices(cfg.symbols, cfg.start, cfg.end, cfg.vendor, cfg.interval)
        df = repair_calendar(df)
        df = engineer_features(df, cfg.features)
        arr = df[cfg.features].dropna().values
        scaler = Scaler(cfg.scaler)
        scaler.fit(arr)
        commit = os.environ.get("GIT_COMMIT", "dev")
        out_dir = Path("data/processed") / commit
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "features.npy", scaler.transform(arr))
        scaler.save(out_dir / "scaler.pkl")
        mlflow.log_artifacts(str(out_dir))


@cli.command()
def build_dataset(config: str = "configs/data.yaml"):
    """Create sliding windows and dataset splits."""
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name=Path(config).name)
    with mlflow.start_run(run_name="data_prep"):
        mlflow.log_params(OmegaConf.to_container(cfg))
        commit = os.environ.get("GIT_COMMIT", "dev")
        proc_dir = Path("data/processed") / commit
        arr = np.load(proc_dir / "features.npy")
        X, y = make_windows(arr, cfg.context_len, cfg.prediction_len)
        X_trainval, y_trainval, X_test, y_test = holdout_split(X, y, cfg.split.test_size_days)
        folds = list(walk_forward_split(X_trainval, y_trainval, cfg.split.n_splits))
        np.savez(proc_dir / "dataset.npz", X_trainval=X_trainval, y_trainval=y_trainval, X_test=X_test, y_test=y_test)
        joblib.dump(folds, proc_dir / "folds.pkl")
        mlflow.log_artifacts(str(proc_dir))


if __name__ == "__main__":
    cli()
