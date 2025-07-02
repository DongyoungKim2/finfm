from __future__ import annotations

import pandas as pd

from src.serving.predictor import Predictor
from .strategy import BaseStrategy


class BacktestEngine:
    def __init__(self, predictor: Predictor, strat: BaseStrategy):
        self.pred = predictor
        self.strat = strat

    def run_slice(self, ctx_df: pd.DataFrame, true_ret: pd.Series):
        est_ret = self.pred.predict_next(ctx_df)
        signal = self.strat.generate(est_ret)
        pl = signal * true_ret.iloc[-1]
        return est_ret, signal, pl

    def walk_forward(self, slices):
        results = []
        for ctx_df, true_ret in slices:
            results.append(self.run_slice(ctx_df, true_ret))
        return pd.DataFrame(results, columns=["est_ret", "signal", "pl"])
