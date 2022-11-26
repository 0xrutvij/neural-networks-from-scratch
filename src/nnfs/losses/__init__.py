__all__ = ["LossFunction"]


from typing import Type

import numpy as np

from .functions import CrossEntropy, SquaredError
from .generic_loss import GenericLoss


class LossFunction:
    def __init__(self, function: Type[GenericLoss], **kwargs) -> None:
        self.fn = function(**kwargs)
        self.backward = self.fn.gradient
        self.accuracy = self.fn.accuracy

    def __call__(self, y: np.ndarray, o: np.ndarray) -> np.ndarray:
        return self.fn.loss(y, o)

    @classmethod
    def squared_error(cls):
        return cls(SquaredError)

    @classmethod
    def cross_entropy(cls):
        return cls(CrossEntropy)
