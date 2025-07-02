"""Evaluation utilities for TimesFM models."""
from __future__ import annotations

import torch


def evaluate(model, dataloader):
    """Placeholder evaluation loop."""
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            results.append(outputs.logits)
    return results
