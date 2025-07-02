"""Prediction utilities for TimesFM models."""
from __future__ import annotations

import json

import joblib
import mlflow.pyfunc
import pandas as pd
import torch

from src.data.fetcher import fetch_prices
from src.data.preprocess import repair_calendar


def fetch_context(
    symbols: list[str],
    end: str | None = None,
    ctx: int = 60,
    vendor: str = "yfinance",
) -> pd.DataFrame:
    """Return a `(ctx, len(symbols) * features)` DataFrame.

    The index is aligned on daily UTC midnight with any gaps forward filled so
    that the returned frame always has exactly ``ctx`` rows.
    """

    if end is None:
        end_dt = pd.Timestamp.utcnow().floor("D") - pd.Timedelta(days=1)
    else:
        end_dt = pd.Timestamp(end).tz_localize("UTC").floor("D")

    start_dt = end_dt - pd.Timedelta(days=ctx * 3)
    df = fetch_prices(
        symbols=symbols,
        start=str(start_dt.date()),
        end=str(end_dt.date()),
        vendor=vendor,
        interval="1d",
    )
    df = repair_calendar(df)
    df = (
        df.reset_index()
        .pivot(index="date", columns="symbol")
        .ffill()
        .tail(ctx)
    )
    df.columns = [f"{sym}_{feat}" for feat, sym in df.columns]
    return df


def merge_lora(model):
    """Merge attached LoRA weights into ``model`` if present."""

    try:  # optional dependency
        from peft import PeftModel
    except Exception:  # pragma: no cover - peft may be missing
        return model

    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    return model


class Predictor:
    """Thin wrapper around TimesFM checkpoints for inference."""

    def __init__(self, mlflow_uri: str, ticker: str):
        self.model, self.meta = self._load_model(mlflow_uri, ticker)
        self.scaler = joblib.load(self.meta["scaler_path"])
        self.ctx_len = self.meta["ctx_len"]
        self.series = self.meta["series_list"]

    def _load_model(self, uri: str, ticker: str):
        m = mlflow.pyfunc.load_model(f"{uri}/TimesFM-PPO/{ticker}/Staging")
        if "lora" in m.metadata.tags:
            merge_lora(m._model)
        meta = json.load(open("meta.json"))
        return m._model.eval(), meta

    def predict_next(self, df_ctx: pd.DataFrame) -> float:
        X = self.scaler.transform(df_ctx[self.series].values).astype("float32")
        tensor = torch.tensor(X).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor).squeeze(0)
        return self._postprocess(out, df_ctx)

    def _postprocess(self, y_pred: torch.Tensor, df_ctx: pd.DataFrame) -> float:
        last_price = df_ctx[f"{self.meta['target_ticker']}_close"].iloc[-1]
        pred_price = last_price * (1 + y_pred.item())
        return float(pred_price)

