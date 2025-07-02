from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def generate(self, est_ret: float) -> int:
        """Return position based on estimated return."""
        raise NotImplementedError


class ThresholdLongShort(BaseStrategy):
    def __init__(self, long: float = 0.002, short: float = -0.002):
        self.long = long
        self.short = short

    def generate(self, est_ret: float) -> int:
        if est_ret > self.long:
            return 1
        if est_ret < self.short:
            return -1
        return 0
