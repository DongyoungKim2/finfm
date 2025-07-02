"""PyTorch Dataset for TimesFM."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TimesFMDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X, self.y = data["X"], data["y"]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])
