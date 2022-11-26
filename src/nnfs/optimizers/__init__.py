from typing import Type

import numpy as np

from .generic_optimizer import GenericOptimizer
from .optimizers import SGD, AdaDelta, AdaGrad, Adam, RMSProp


class Optimizer:
    def __init__(self, optimizer: Type[GenericOptimizer], **kwargs) -> None:
        self.optimizer = optimizer
        self.kwargs = kwargs
        self.id_cache: dict[int, GenericOptimizer] = dict()

    def __call__(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        return self.update(w, dw)

    def update(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        if (wid := id(w)) not in self.id_cache:
            self.id_cache[wid] = self.optimizer(**self.kwargs)
        return self.id_cache[wid].update(w, dw)

    @classmethod
    def sgd(cls, momentum: float = 0.4, learning_rate: float = 0.001):
        return cls(SGD, learning_rate=learning_rate, momentum=momentum)

    @classmethod
    def adagrad(cls, learning_rate: float = 0.001):
        return cls(AdaGrad, learning_rate=learning_rate)

    @classmethod
    def adadelta(cls, rho: float = 0.95, eps: float = 1e-6):
        return cls(AdaDelta, rho=rho, eps=eps)

    @classmethod
    def rmsprop(cls, learning_rate: float = 0.001, rho: float = 0.9):
        return cls(RMSProp, learning_rate=learning_rate, rho=rho)

    @classmethod
    def adam(cls, learning_rate: float = 0.0001, b1: float = 0.9, b2: float = 0.999):
        return cls(Adam, learning_rate=learning_rate, b1=b1, b2=b2)
