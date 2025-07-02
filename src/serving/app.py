"""FastAPI service for TimesFM predictions."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.serving.predictor import predict

app = FastAPI(title="FinFM TimesFM Service")


class PredictRequest(BaseModel):
    symbol: str
    context: list[float]
    horizon: int = 1


@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    result = predict(req.symbol, req.context, req.horizon)
    return {"predictions": result}
