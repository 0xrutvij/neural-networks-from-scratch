from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

S = TypeVar("S")
T = TypeVar("T", bound=np.dtype)


class GenericOptimizer(ABC):
    def __init__(self, learning_rate: float = 0.001, **kwargs) -> None:
        super().__init__()
        self.eta = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray[S, T], dw: np.ndarray[S, T]) -> np.ndarray[S, T]:
        pass
