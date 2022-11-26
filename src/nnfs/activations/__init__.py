from typing import Type

import numpy as np

from .functions import ELU, Identity, LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from .generic_activation import GenericActivation

__all__ = ["Activation"]


class Activation:
    def __init__(self, function: Type[GenericActivation], **kwargs):
        self.fn = function(**kwargs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return self.fn.gradient(x)

    @classmethod
    def relu(cls):
        return cls(ReLU)

    @classmethod
    def leakyRelu(cls, alpha: float = 0.2):
        return cls(LeakyReLU, alpha=alpha)

    @classmethod
    def sigmoid(cls):
        return cls(Sigmoid)

    @classmethod
    def tanh(cls):
        return cls(Tanh)

    @classmethod
    def softmax(cls):
        return cls(Softmax)

    @classmethod
    def elu(cls, alpha: float = 0.2):
        return cls(ELU, alpha=alpha)

    @classmethod
    def identity(cls):
        return cls(Identity)
