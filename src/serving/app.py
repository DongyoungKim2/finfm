"""FastAPI service exposing TimesFM predictions."""
from __future__ import annotations

import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.serving.predictor import Predictor, fetch_context


MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow.internal:5000")

app = FastAPI()


@app.on_event("startup")
def load() -> None:  # pragma: no cover - simple side effect
    global predictor
    predictor = Predictor(MLFLOW_URI, ticker=os.getenv("TICKER", "AAPL"))


class Input(BaseModel):
    context: list[list[float]] | None = None


@app.post("/predict")
def predict(inp: Input | None = None):
    if inp and inp.context:
        df_ctx = pd.DataFrame(inp.context, columns=predictor.series)
    else:
        df_ctx = fetch_context(predictor.series, ctx=predictor.ctx_len)
    price = predictor.predict_next(df_ctx)
    return {
        "ticker": predictor.meta["target_ticker"],
        "predicted_price": price,
    }

