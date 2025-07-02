"""Preprocessing and feature engineering pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .calendar import trading_days


FEATURE_FUNCS = {
    "returns": lambda df: df["close"].pct_change(),
    "logret": lambda df: np.log(df["close"]).diff(),
    "volatility_20": lambda df: np.log(df["close"]).diff().rolling(20).std(),
    "sma_10": lambda df: df["close"].rolling(10).mean(),
    "sma_30": lambda df: df["close"].rolling(30).mean(),
    "ema_12": lambda df: df["close"].ewm(span=12).mean(),
    "ema_26": lambda df: df["close"].ewm(span=26).mean(),
}


def repair_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing trading days for each symbol."""
    all_frames = []
    for symbol, g in df.groupby(level=1):
        start, end = g.index.get_level_values(0)[[0, -1]]
        cal = trading_days(str(start.date()), str(end.date()))
        g = g.droplevel(1).reindex(cal).ffill()
        g["symbol"] = symbol
        all_frames.append(g)
    result = pd.concat(all_frames)
    result.index.name = "date"
    result = result.reset_index().set_index(["date", "symbol"]).sort_index()
    return result


def engineer_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Compute the selected technical indicators.

    Parameters
    ----------
    df:
        Price DataFrame indexed by ``date`` and ``symbol``. Some users may load
        prices with a multi-indexed column layout (for example when downloading
        several tickers at once with ``yfinance``).  To make the feature
        engineering robust we detect this case and reshape the frame to the
        expected long format before computing the indicators.
    features:
        Iterable of feature names to compute.  Only names present in
        ``FEATURE_FUNCS`` are applied.
    """

    out = df.copy()

    # When the frame has multi-index columns (typically ticker names as one
    # level) the indicator functions expect a long format with ``symbol`` as an
    # index level and simple OHLCV columns.  Always stack the last column level
    # to achieve this; if the original frame already carried a ``symbol`` index
    # we drop it to avoid duplicated levels.
    if isinstance(out.columns, pd.MultiIndex):
        out = out.stack()
        if "symbol" in df.index.names:
            out = out.droplevel(1)
        out.index.names = ["date", "symbol"]
        out = out.sort_index()

    for feat in features:
        func = FEATURE_FUNCS.get(feat)
        if func is not None:
            out[feat] = func(out)

    return out
