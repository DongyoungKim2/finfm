"""Scalers with persistence support."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler as _SS, MinMaxScaler as _MM


@dataclass
class Scaler:
    kind: Literal["standard", "minmax"] = "standard"
    scaler: _SS | _MM | None = None

    def fit(self, arr: np.ndarray) -> None:
        if self.kind == "standard":
            self.scaler = _SS()
        else:
            self.scaler = _MM()
        self.scaler.fit(arr)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        assert self.scaler is not None
        return self.scaler.transform(arr)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        assert self.scaler is not None
        return self.scaler.inverse_transform(arr)

    def save(self, path: str | Path) -> None:
        assert self.scaler is not None
        joblib.dump({"kind": self.kind, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        data = joblib.load(path)
        obj = cls(kind=data["kind"])
        obj.scaler = data["scaler"]
        return obj
