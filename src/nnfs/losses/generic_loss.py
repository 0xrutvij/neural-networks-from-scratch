from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
S = TypeVar("S", bound=np.dtype)


class GenericLoss(ABC):
    """A Generic loss function interface"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def loss(self, y: np.ndarray[T, S], o: np.ndarray[T, S]) -> np.ndarray[T, S]:
        pass

    @abstractmethod
    def gradient(self, y: np.ndarray[T, S], o: np.ndarray[T, S]) -> np.ndarray[T, S]:
        pass

    @abstractmethod
    def accuracy(self, y: np.ndarray[T, S], o: np.ndarray[T, S]) -> Any:
        pass
