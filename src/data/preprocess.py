"""Preprocessing utilities for TimesFM."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def preprocess(input_path: str, output_dir: str) -> None:
    """Simple example preprocessing that saves numpy arrays."""
    df = pd.read_csv(input_path, parse_dates=["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ticker, group in df.groupby("Ticker"):
        out_file = Path(output_dir) / f"{ticker}.csv"
        group.to_csv(out_file, index=False)
