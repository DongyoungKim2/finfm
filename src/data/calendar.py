"""Trading-day calendar utilities."""
from __future__ import annotations

import pandas as pd


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return a business-day index for the given range."""
    return pd.bdate_range(start=start, end=end, freq="C")
