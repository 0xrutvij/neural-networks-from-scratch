"""
Reference Links:
    - https://www.v7labs.com/blog/neural-networks-activation-functions#h4
    - https://en.wikipedia.org/wiki/Activation_function
    - https://blog.paperspace.com/vanishing-gradients-activation-function/
"""
from typing import final

from .generic_activation import GenericActivation, np


@final
class Identity(GenericActivation):
    """Identity Activation - Equivalent to No-Op"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1]])


@final
class Sigmoid(GenericActivation):
    """Sigmoid (Logistic) Activation Function
    Properties:
        - Range from [0, 1]

    Advantages:
        - Useful for probabilities
        - Differentiable, w/ a smooth gradient

    Limitations:
        - Gradients are significant only in a narrow range,
        [-3, 3] and thus is susceptible to vanishing gradients.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (1 + np.exp(-x)) ** -1

    def gradient(self, x: np.ndarray) -> np.ndarray:
        p = self(x)
        return p * (1 - p)


@final
class Tanh(GenericActivation):
    """Hyperbolic Tangent Activation Funciton
    Properties:
        - Range from [-1, 1]

    Advantages:
        - Zero-centred
        - Mean output for a neuron is thus close to 0 and
        simplifies learning for following layers.

    Limitations:
        - Also susceptible to vanishing gradients like
        the sigmoid function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 - self(x) ** 2


@final
class ReLU(GenericActivation):
    """Rectified Linear Unit
    Properties:
        - Linear, non-saturating

    Advantages:
        - Computationally efficient
        - Accelerates convergence to global minimum

    Limitations:
        - Dying ReLU: for negative inputs, the gradient is zero
        and thus no weight updates might occur causing dead neurons.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


@final
class LeakyReLU(GenericActivation):
    """Leaky Rectified Linear Unit
    Properties:
        - Linear, non-saturating
        - Non-zero gradients for x < 0

    Advantages:
        - Computationally efficient
        - Accelerates convergence to global minimum

    Limitations:
        - Inconsistent predictions for negative inputs
        - Learning for negative values is slow due to a small
        gradient.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


@final
class ELU(GenericActivation):
    """Exponential Linear Unit
    Properties:
        - Non-zero gradients for x < 0

    Advantages:
        - Slow smoothing for negative values

    Limitations:
        - Computationally intensive
        - Susceptible to exploding gradients
    """

    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self(x) + self.alpha)


@final
class Softmax(GenericActivation):
    """Softmax Activation Funciton
    Properties:
        - Non-zero gradients for x < 0

    Advantages:
        - Slow smoothing for negative values

    Limitations:
        - Computationally intensive
        - Susceptible to exploding gradients
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # take delta with max for numerical stability
        e_dx = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_dx / np.sum(e_dx, axis=-1, keepdims=True)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        p = self(x)
        return p * (1 - p)
