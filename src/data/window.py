"""Sliding-window creation using stride tricks."""
from __future__ import annotations

import numpy as np


def make_windows(arr: np.ndarray, ctx: int, pred: int):
    """Return context and prediction windows."""
    T, F = arr.shape
    view = np.lib.stride_tricks.sliding_window_view(arr, (ctx + pred), axis=0)
    view = view.reshape(-1, ctx + pred, F)
    return view[:, :ctx], view[:, ctx:]
