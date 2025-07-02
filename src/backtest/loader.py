from __future__ import annotations

import pandas as pd
from typing import Iterable

from src.data.fetcher import fetch_prices
from src.data.preprocess import repair_calendar


def get_walk_forward_slices(
    meta: dict,
    start: str,
    end: str,
    train_span_days: int,
    test_span_days: int,
    vendor: str = "yfinance",
) -> list[tuple[pd.DataFrame, pd.Series]]:
    """Return expanding train/rolling test slices of context and true returns."""

    symbols = meta.get("series_list", [])
    df = fetch_prices(symbols, start, end, vendor=vendor, interval="1d")
    df = repair_calendar(df)
    df = (
        df.reset_index()
        .pivot(index="date", columns="symbol")
        .ffill()
    )
    df.columns = [f"{sym}_{col}" for col, sym in df.columns]

    logret = df[f"{meta['target_ticker']}_close"].pct_change().apply(lambda x: 0.0 if pd.isna(x) else x)
    df["_logret"] = logret

    dates = df.index
    slices: list[tuple[pd.DataFrame, pd.Series]] = []
    start_idx = 0
    while True:
        train_end = start_idx + train_span_days
        test_end = train_end + test_span_days
        if test_end > len(df):
            break
        ctx_df = df.iloc[train_end - meta["ctx_len"]:train_end]
        true_returns = df["_logret"].iloc[train_end:test_end]
        slices.append((ctx_df, true_returns))
        start_idx += test_span_days
    return slices
