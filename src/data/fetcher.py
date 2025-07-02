"""Data fetching utilities using yfinance."""
from __future__ import annotations

from typing import Iterable
import pandas as pd
import yfinance as yf


def download(tickers: Iterable[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download historical data for the given tickers."""
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )
    frames = []
    for ticker in tickers:
        df = data[ticker].copy()
        df["Ticker"] = ticker
        df.reset_index(inplace=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
