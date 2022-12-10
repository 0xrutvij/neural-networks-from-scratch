from math import sqrt
from typing import Annotated, Optional, final

import numpy as np
from numpy._typing import _Shape

if __name__ == "__main__":
    from nnfs.activations import Activation
    from nnfs.layers.generic_layer import GenericLayer
    from nnfs.optimizers import Optimizer
else:
    from ..activations import Activation
    from ..optimizers import Optimizer
    from .generic_layer import GenericLayer


@final
class Dense(GenericLayer):
    """Fully-connected NN layer."""

    def __init__(self, n_neurons: int, shape: _Shape, optimizer: Optimizer) -> None:
        super().__init__(shape)
        self.n_neurons = n_neurons
        self.trainable = True
        self.weights, self.bias = self._initialize(n_neurons, shape)
        self.optimizer = optimizer
        self.last_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        self.last_input = x
        return x.dot(self.weights) + self.bias

    def backward(self, accum_grad: np.ndarray) -> np.ndarray:
        weights = self.weights

        if (
            self.weights is not None
            and self.bias is not None
            and self.last_input is not None
        ):

            if self.trainable:

                d_weights = self.last_input.T.dot(accum_grad)
                d_bias = np.sum(accum_grad, axis=0, keepdims=True)

                self.weights = self.optimizer.update(self.weights, d_weights)
                self.bias = self.optimizer.update(self.bias, d_bias)

            accum_grad = accum_grad.dot(weights.T)

        return accum_grad

    def n_parameters(self) -> int:
        return int(np.prod(self.weights.shape) + np.prod(self.bias.shape))

    def output_shape(self) -> _Shape:
        return (self.input_shape[0], self.n_neurons)

    @staticmethod
    def _initialize(
        n_neurons: int,
        shape: _Shape,
        scale_factor: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = 1 / (sqrt(shape[0]) * scale_factor)
        weights = np.random.uniform(-limit, limit, (shape[1], n_neurons))
        bias = np.zeros((1, n_neurons))
        return weights, bias


@final
class BatchNorm(GenericLayer):
    """Batch normalization layer"""

    def __init__(
        self, shape: _Shape, optimizer: Optimizer, monemntum: float = 99e-2
    ) -> None:
        super().__init__(shape)
        self.momentum = monemntum
        self.optimizer = optimizer
        self.trainable = True
        self.epsilon = 1e-2
        self.first_pass = True
        (
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_variance,
            self.x_centered,
            self.stddev_inv,
        ) = self._initalize(shape)

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        kwargs = {"a": x, "axis": 0, "keepdims": True}

        if self.first_pass:
            self.running_mean = np.mean(**kwargs)  # type: ignore
            self.running_variance = np.mean(**kwargs)  # type: ignore
            self.first_pass = False

        mean, variance = self.running_mean, self.running_variance

        if training and self.trainable:
            mean, variance = map(
                lambda f: f(**kwargs), (np.mean, np.var)  # type: ignore
            )
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )

            self.running_variance = (
                self.momentum * self.running_variance + (1 - self.momentum) * variance
            )

        self.x_centered = x - mean
        self.stddev_inv = 1 / np.sqrt(variance + self.epsilon)
        x_normalized = self.x_centered * self.stddev_inv
        output = self.gamma * x_normalized + self.beta

        return output

    def backward(self, accum_grad: np.ndarray) -> np.ndarray:

        gamma = self.gamma

        if self.trainable:
            x_normalized = self.x_centered * self.stddev_inv
            d_gamma = np.sum(accum_grad * x_normalized, axis=0)
            d_beta = np.sum(accum_grad, axis=0)

            self.gamma = self.optimizer.update(self.gamma, d_gamma)
            self.beta = self.optimizer.update(self.beta, d_beta)

        batch_size = accum_grad.shape[0]

        coeff = (1 / batch_size) * gamma * self.stddev_inv
        updated_grad = (
            batch_size * accum_grad
            - np.sum(accum_grad, axis=0)
            - self.x_centered
            * self.stddev_inv**2
            * np.sum(accum_grad * self.x_centered, axis=0)
        )

        return coeff * updated_grad

    def n_parameters(self) -> int:
        return int(sum(map(np.prod, (self.gamma.shape, self.beta.shape))))

    def output_shape(self) -> _Shape:
        return self.input_shape

    @staticmethod
    def _initalize(shape: _Shape) -> Annotated[tuple[np.ndarray, ...], 6]:
        gamma, beta = map(lambda f: f((1, shape[1])), (np.ones, np.zeros))
        mean, variance, x_centered, stddev_inv = (
            np.zeros((1, shape[1])) for _ in range(4)
        )
        return gamma, beta, mean, variance, x_centered, stddev_inv


@final
class Reshape(GenericLayer):
    """Used to reshape input tensors"""

    def __init__(self, shape: _Shape, output_shape: _Shape) -> None:
        super().__init__(shape)
        self.transformed_shape = output_shape

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        return x.reshape(self.transformed_shape)

    def backward(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad.reshape(self.input_shape)

    def n_parameters(self) -> int:
        return 0

    def output_shape(self) -> _Shape:
        return self.transformed_shape


@final
class Dropout(GenericLayer):
    def __init__(self, shape: _Shape, p_drop: float = 2e-1) -> None:
        super().__init__(shape)
        self.probability_of_drop = p_drop
        self.n_neurons = None
        self.pass_through = True
        self._mask = np.ones_like(self.input_shape)

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        if training:
            self._mask = np.random.uniform(size=x.shape) > self.probability_of_drop
        else:
            self._mask = np.ones_like(x)
        return x * self._mask

    def backward(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad * self._mask

    def n_parameters(self) -> int:
        return 0

    def output_shape(self) -> _Shape:
        return self.input_shape


class ActivationLayer(GenericLayer):
    """A layers to encompass activation functions"""

    def __init__(self, shape: _Shape, activation: Activation) -> None:
        super().__init__(shape)
        self.activation = activation
        self.last_input = np.zeros(shape)
        self.layer_name = f"Activation {self.activation.fn.__class__.__name__}"

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        self.last_input = x
        return self.activation.forward(x)

    def backward(self, accum_grad: np.ndarray) -> np.ndarray:
        return accum_grad * self.activation.backward(self.last_input)

    def n_parameters(self) -> int:
        return 0

    def output_shape(self) -> _Shape:
        return self.input_shape
