"""Utilities to download and cache raw price data."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Iterable, List

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

RAW_DIR = Path("data/raw")


def _cache_path(vendor: str, symbol: str) -> Path:
    path = RAW_DIR / vendor / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@retry(stop=stop_after_attempt(3), wait=wait_fixed(15))
def _fetch_alpha_vantage(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Alpha Vantage respecting rate limits."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if api_key is None:
        raise RuntimeError("ALPHAVANTAGE_API_KEY environment variable not set")

    if interval == "1d":
        function = "TIME_SERIES_DAILY_ADJUSTED"
        url = (
            f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv"
        )
    else:
        function = "TIME_SERIES_INTRADAY"
        url = (
            f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize=full&apikey={api_key}&datatype=csv"
        )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(resp.text), parse_dates=[0])
    df.rename(
        columns={
            "timestamp": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjusted_close": "close",
            "volume": "volume",
        },
        inplace=True,
    )
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC").dt.floor("D")
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[start:end]
    return df


def _fetch_yfinance(symbol: str, start: str, end: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=auto_adjust, threads=True)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize("UTC").floor("D")
    return df


def fetch_prices(
    symbols: List[str],
    start: str,
    end: str,
    vendor: Literal["yfinance", "alphavantage"],
    interval: Literal["1d", "1h"],
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download prices and return a multi-indexed DataFrame."""
    frames = []
    for symbol in symbols:
        cache = _cache_path(vendor, symbol)
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            if vendor == "yfinance":
                df = _fetch_yfinance(symbol, start, end, interval, auto_adjust)
            else:
                df = _fetch_alpha_vantage(symbol, start, end, interval)
            df.to_parquet(cache, compression="zstd", row_group_size=100_000)
        df = df.loc[start:end]
        df["symbol"] = symbol
        frames.append(df)

    all_df = pd.concat(frames)
    all_df.index.name = "date"
    all_df.sort_index(inplace=True)
    all_df = all_df.reset_index().set_index(["date", "symbol"]).sort_index()
    return all_df
