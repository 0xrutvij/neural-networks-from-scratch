from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from numpy._typing import _Shape

T = TypeVar("T")
S = TypeVar("S", bound=np.dtype)


class GenericLayer(ABC):
    def __init__(self, shape: _Shape) -> None:
        self._input_shape = shape
        self.layer_name = self.__class__.__name__

    @property
    def input_shape(self) -> _Shape:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape: _Shape):
        self._input_shape = shape

    @abstractmethod
    def n_parameters(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: np.ndarray[T, S], training: bool) -> np.ndarray[T, S]:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, accum_grad: np.ndarray[T, S]) -> np.ndarray[T, S]:
        raise NotImplementedError()

    @abstractmethod
    def output_shape(self) -> _Shape:
        raise NotImplementedError()
