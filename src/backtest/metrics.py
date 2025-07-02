from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def trading_stats(pl_series: pd.Series, initial_cash: float = 1.0) -> dict:
    try:
        import vectorbt as vbt  # type: ignore

        returns = pl_series / initial_cash
        pf = vbt.Portfolio.from_returns(returns)
        return {
            "cum_return": float(pf.total_return()),
            "sharpe": float(pf.sharpe_ratio()),
            "max_dd": float(pf.max_drawdown()),
        }
    except Exception:
        # Fallback simple metrics
        equity = (pl_series / initial_cash + 1).cumprod()
        cum_return = equity.iloc[-1] - 1
        ret = pl_series / initial_cash
        sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(len(ret))
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        return {"cum_return": float(cum_return), "sharpe": float(sharpe), "max_dd": float(max_dd)}


def compute_all_metrics(df_res: pd.DataFrame) -> dict:
    y_true = df_res["pl"].values
    y_pred = df_res["est_ret"].values
    out = {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    out.update(trading_stats(df_res["pl"]))
    return out
