"""Batch prediction script for cron jobs."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import os

import click
import pandas as pd

from src.serving.predictor import Predictor, fetch_context


MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow.internal:5000")


@click.command()
@click.option("--ticker", required=True)
@click.option("--horizon", default=1)
def run_batch(ticker: str, horizon: int) -> None:
    """Generate a prediction and save it to ``data/predictions``."""

    pred = Predictor(MLFLOW_URI, ticker)
    ctx = fetch_context(pred.series, ctx=pred.ctx_len)
    price = pred.predict_next(ctx)

    out_path = Path("data/predictions") / f"{date.today()}_{ticker}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pred_price": [price]}, index=[pd.Timestamp.today()]).to_csv(out_path)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    run_batch()

