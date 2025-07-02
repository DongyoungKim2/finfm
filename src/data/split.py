"""Dataset split utilities."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_split(X: np.ndarray, y: np.ndarray, n_splits: int) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tss.split(X):
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def holdout_split(X: np.ndarray, y: np.ndarray, test_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split off the last *test_size* samples as hold-out."""
    split = len(X) - test_size
    return X[:split], y[:split], X[split:], y[split:]
