"""
Reference Links:
    - https://arxiv.org/pdf/1609.04747.pdf
    - https://ruder.io/optimizing-gradient-descent/index.html
"""
from typing import Optional, final

import numpy as np

from .generic_optimizer import GenericOptimizer


@final
class SGD(GenericOptimizer):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.4) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.w_update: Optional[np.ndarray] = None

    def update(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        if self.w_update is None:
            self.w_update = np.zeros_like(w)

        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * dw

        w -= self.eta * self.w_update
        return w


@final
class AdaGrad(GenericOptimizer):
    def __init__(self, learning_rate: float = 0.001) -> None:
        super().__init__(learning_rate)
        self.sq_sum_gradients: Optional[np.ndarray] = None
        self.epsilon = 1e-8

    def update(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        if self.sq_sum_gradients is None:
            self.sq_sum_gradients = np.full_like(w, self.epsilon)

        self.sq_sum_gradients += dw**2

        w -= self.eta * dw / np.sqrt(self.sq_sum_gradients + self.epsilon)
        return w


@final
class AdaDelta(GenericOptimizer):
    def __init__(self, rho: float = 0.95, eps: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = eps
        self.rho = rho

        self.weight_update: Optional[np.ndarray] = None
        self.mean_sq_weight_update: Optional[np.ndarray] = None
        self.mean_sq_dw: Optional[np.ndarray] = None

    def update(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        if self.weight_update is None:
            self.weight_update = np.zeros_like(w)

        if self.mean_sq_weight_update is None:
            self.mean_sq_weight_update = np.zeros_like(w)

        if self.mean_sq_dw is None:
            self.mean_sq_dw = np.zeros_like(dw)

        self.mean_sq_dw = self.rho * self.mean_sq_dw + (1 - self.rho) * dw**2

        rms_weight_update = np.sqrt(self.mean_sq_weight_update + self.epsilon)
        rms_dw = np.sqrt(self.mean_sq_dw + self.epsilon)

        # adaptive learning rate
        self.eta = rms_weight_update / rms_dw

        self.weight_update = self.eta * dw

        self.mean_sq_weight_update = (
            self.rho * self.mean_sq_weight_update
            + (1 - self.rho) * self.weight_update**2
        )

        w -= self.weight_update
        return w


@final
class RMSProp(GenericOptimizer):
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9) -> None:
        super().__init__(learning_rate)
        self.mean_sq_dw: Optional[np.ndarray] = None
        self.epsilon = 1e-8
        self.rho = rho

    def update(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:

        if self.mean_sq_dw is None:
            self.mean_sq_dw = np.zeros_like(dw)

        self.mean_sq_dw = self.rho * self.mean_sq_dw + (1 - self.rho) * dw**2

        w -= self.eta * dw / np.sqrt(self.mean_sq_dw + self.epsilon)
        return w


@final
class Adam(GenericOptimizer):
    def __init__(
        self, learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999
    ) -> None:
        super().__init__(learning_rate)
        self.epsilon = 1e-8
        self.wb1: Optional[np.ndarray] = None
        self.wb2: Optional[np.ndarray] = None
        self.weight_update: Optional[np.ndarray] = None
        # rates of decay
        self.b1, self.b2 = b1, b2

    def update(self, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        if self.wb1 is None:
            self.wb1 = np.zeros_like(dw)

        if self.wb2 is None:
            self.wb2 = np.zeros_like(dw)

        if self.weight_update is None:
            self.weight_update = np.zeros_like(w)

        self.wb1 = self.b1 * self.wb1 + (1 - self.b1) * dw
        self.wb2 = self.b2 * self.wb2 + (1 - self.b2) * dw**2

        wb1_hat = self.wb1 / (1 - self.b1)
        wb2_hat = self.wb2 / (1 - self.b2)

        self.weight_update = self.eta * wb1_hat / (np.sqrt(wb2_hat) + self.epsilon)

        w -= self.weight_update
        return w
