from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from src.serving import app as serving_app
from src.serving.predictor import Predictor


def _dummy_predictor(tmp_path) -> Predictor:
    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    class DummyModel(torch.nn.Module):
        def forward(self, x):  # type: ignore
            return torch.tensor([[0.1]], dtype=torch.float32)

    class DummyPredictor(Predictor):
        def _load_model(self, uri: str, ticker: str):
            meta = {
                "scaler_path": str(scaler_path),
                "ctx_len": 2,
                "series_list": ["a", "b"],
                "target_ticker": "a",
            }
            return DummyModel(), meta

    return DummyPredictor("uri", "TST")


def test_predict_next_returns_float(tmp_path):
    pred = _dummy_predictor(tmp_path)
    df = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], columns=pred.series)
    out = pred.predict_next(df)
    assert isinstance(out, float)
    assert not np.isnan(out)


def test_fastapi_predict(monkeypatch, tmp_path):
    pred = _dummy_predictor(tmp_path)

    def _load():
        serving_app.predictor = pred

    monkeypatch.setattr(serving_app, "load", _load)
    serving_app.load()

    client = TestClient(serving_app.app)
    resp = client.post("/predict", json={"context": [[1.0, 2.0], [1.0, 2.0]]})
    assert resp.status_code == 200
    assert "predicted_price" in resp.json()

