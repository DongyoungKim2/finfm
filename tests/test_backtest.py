from __future__ import annotations

import pandas as pd
import numpy as np

from src.backtest.strategy import ThresholdLongShort
from src.backtest.engine import BacktestEngine
from src.serving.predictor import Predictor


class DummyPredictor(Predictor):
    def _load_model(self, uri: str, ticker: str):
        meta = {
            "scaler_path": "",
            "ctx_len": 2,
            "series_list": ["a", "b"],
            "target_ticker": "a",
        }

        class DummyModel:
            def __call__(self, x):
                return np.array([[0.01]], dtype=np.float32)

        return DummyModel(), meta

    def predict_next(self, df_ctx: pd.DataFrame) -> float:  # type: ignore[override]
        return 0.01


def test_threshold_generate():
    strat = ThresholdLongShort(long=0.005, short=-0.005)
    assert strat.generate(0.01) == 1
    assert strat.generate(-0.01) == -1
    assert strat.generate(0.0) == 0


def test_engine_walk_forward():
    pred = DummyPredictor("uri", "TST")
    strat = ThresholdLongShort()
    engine = BacktestEngine(pred, strat)
    ctx = pd.DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    true_ret = pd.Series([0.02])
    est, sig, pl = engine.run_slice(ctx, true_ret)
    assert est == 0.01
    assert sig in (-1, 0, 1)
    assert isinstance(pl, float)
