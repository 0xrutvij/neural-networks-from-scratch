from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

T = TypeVar("T")
S = TypeVar("S", bound=np.dtype)


class GenericActivation(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: np.ndarray[T, S]) -> np.ndarray[T, S]:
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray[T, S]) -> np.ndarray[T, S]:
        pass
