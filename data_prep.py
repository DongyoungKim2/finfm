"""Utilities for downloading and preparing historical price data.

This script lists a set of major U.S. stocks and ETFs and downloads
historical price data using the ``yfinance`` package. The data can then
be used to train or fine-tune ``timesfm`` models and to update datasets
for future predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf

MAJOR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "BRK-B", "JPM", "V",
    "JNJ", "UNH", "XOM", "LLY", "BAC",
]

MAJOR_ETFS = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    "VOO", "IVV", "AGG", "VUG", "VTV",
    "XLK", "XLF", "XLV",
]


def get_major_tickers() -> List[str]:
    """Return a list of major U.S. stock and ETF tickers."""
    return sorted(MAJOR_STOCKS + MAJOR_ETFS)


def download_history(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical OHLCV data for *tickers* using yfinance.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    start:
        Start date string in ``YYYY-MM-DD`` format.
    end:
        End date string in ``YYYY-MM-DD`` format.
    interval:
        Data interval. See ``yfinance`` documentation (e.g. "1d", "1wk").
    """
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


def append_latest(csv_path: Path, interval: str = "1d") -> None:
    """Append the latest data to an existing CSV file.

    The function infers the set of tickers and last date from the file
    and downloads any missing data.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    existing = pd.read_csv(csv_path, parse_dates=["Date"])
    tickers = existing["Ticker"].unique()
    last_date = existing["Date"].max()
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.utcnow().strftime("%Y-%m-%d")

    new_data = download_history(tickers, start=start, end=end, interval=interval)
    if new_data.empty:
        print("Data is already up to date.")
        return

    updated = pd.concat([existing, new_data], ignore_index=True)
    updated.sort_values(["Ticker", "Date"], inplace=True)
    updated.to_csv(csv_path, index=False)
    print(f"Appended {len(new_data)} rows to {csv_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-tickers", action="store_true", help="List major tickers")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    parser.add_argument("--output", help="Path to output CSV")
    parser.add_argument("--update", help="Path to existing CSV to update")

    args = parser.parse_args(argv)

    if args.list_tickers:
        print("\n".join(get_major_tickers()))
        return

    if args.update:
        append_latest(Path(args.update), interval=args.interval)
        return

    if not args.output:
        parser.error("--output is required when downloading data")

    if not args.start or not args.end:
        parser.error("--start and --end dates are required")

    data = download_history(
        get_major_tickers(),
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    data.sort_values(["Ticker", "Date"], inplace=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
