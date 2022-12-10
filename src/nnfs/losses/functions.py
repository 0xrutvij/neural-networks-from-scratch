from typing import Any, final

import numpy as np

from .generic_loss import GenericLoss


@final
class SquaredError(GenericLoss):
    def loss(self, y: np.ndarray, o: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(y - o)

    def gradient(self, y: np.ndarray, o: np.ndarray) -> np.ndarray:
        return -(y - o)

    def accuracy(self, y: np.ndarray, o: np.ndarray) -> float:
        """R-Squared/Coeff of Determination"""
        y_mean = np.mean(y, axis=0, keepdims=True)
        # ss residuals
        sq_sum_errors: float = np.sum(np.square(y - o))
        # ss total
        sq_sum_diff_avg: float = np.sum(np.square(y - y_mean))
        return np.round(1 - (sq_sum_errors / sq_sum_diff_avg), 4)


@final
class CrossEntropy(GenericLoss):
    def loss(self, y: np.ndarray, o: np.ndarray) -> np.ndarray:
        lower_limit = 1e-15
        o = np.clip(o, lower_limit, 1 - lower_limit)
        return -y * np.log(o) - (1 - y) * np.log(1 - o)

    def gradient(self, y: np.ndarray, o: np.ndarray) -> np.ndarray:
        lower_limit = 1e-15
        o = np.clip(o, lower_limit, 1 - lower_limit)
        return -(y / o) + (1 - y) / (1 - o)

    def accuracy(self, y: np.ndarray, o: np.ndarray) -> Any:
        raise NotImplementedError
