"""Model prediction utilities."""
from __future__ import annotations

from typing import List

import torch

from src.model.timesfm_wrapper import load_model


_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model()
        _model.eval()
    return _model, _tokenizer


def predict(symbol: str, context: List[float], horizon: int = 1) -> List[float]:
    model, tokenizer = get_model()
    # Dummy implementation that returns zeros
    return [0.0 for _ in range(horizon)]
