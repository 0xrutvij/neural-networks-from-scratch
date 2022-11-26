__all__ = ["LossFunction"]


from typing import Type

from .functions import CrossEntropy, SquaredError
from .generic_loss import GenericLoss


class LossFunction:
    def __init__(self, function: Type[GenericLoss], **kwargs) -> None:
        self.fn = function(**kwargs)
        self.__call__ = self.fn.loss
        self.backward = self.fn.gradient
        self.accuracy = self.fn.accuracy

    @classmethod
    def squared_error(cls):
        cls(SquaredError)

    @classmethod
    def cross_entropy(cls):
        cls(CrossEntropy)
