from dataclasses import dataclass, field
from time import sleep
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from numpy._typing import _Shape

from ..layers import GenericLayer
from ..losses import LossFunction
from ..optimizers import Optimizer
from ..utils import Tools


@dataclass
class Errors:
    training: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Training Errors: {list(map(lambda x: round(x, 4), self.training))}\n"
            f"Validation Errors: {list(map(lambda x: round(x, 4), self.validation))}\n"
        )


class NerualNetwork:
    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: LossFunction,
        validation_data: Optional[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.optimizer = optimizer
        self.layers: list[GenericLayer] = []
        self.errors = Errors()
        self.loss_function = loss_fn

        self.validation_x, self.validation_y = None, None
        if validation_data is not None:
            self.validation_x, self.validation_y = validation_data

    def add(self, layer: GenericLayer) -> _Shape:
        self.layers.append(layer)
        return self.layers[-1].output_shape()

    def _batch_pass(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, Any, np.ndarray]:
        y_pred = self._forward_pass(x)
        loss = float(np.mean(self.loss_function(y, y_pred)))
        accuracy = self.loss_function.accuracy(y, y_pred)
        return loss, accuracy, y_pred

    def batch_test(self, x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
        loss, accuracy, _ = self._batch_pass(x, y)
        return loss, accuracy

    def batch_train(self, x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
        loss, accuracy, y_pred = self._batch_pass(x, y)
        loss_grad = self.loss_function.backward(y, y_pred)
        self._backward_pass(loss_grad)
        return loss, accuracy

    def _forward_pass(self, x: np.ndarray, training: bool = True) -> np.ndarray:

        layer_output = x
        for layer in self.layers:
            # print(layer_output.shape)
            layer_output = layer.forward(layer_output, training)

        # print(layer_output.shape)
        return layer_output

    def _backward_pass(self, loss_grad: np.ndarray):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def _batch_fit(
        self, x: np.ndarray, y: np.ndarray, batch_size: int, validation: bool
    ):
        batch_error = []
        for bx, by in Tools.batch_generator(x, y, batch_size):
            loss, _ = self.batch_train(bx, by)
            batch_error.append(loss)

        self.errors.training.append(float(np.mean(batch_error)))

        if validation:
            val_loss, _ = self.batch_test(
                self.validation_x, self.validation_y  # type: ignore
            )
            self.errors.validation.append(float(val_loss))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        batch_size: int,
        live_update: bool = False,
        interval: int = 1,
    ):

        if interval == 1:
            interval = int(n_epochs * 0.1)

        n_epochs += interval

        validation = isinstance(self.validation_x, np.ndarray) and isinstance(
            self.validation_y, np.ndarray
        )

        for i in range(n_epochs):
            self._batch_fit(x, y, batch_size, validation)
            if (i + 1) % interval == 0 and live_update:
                clear_output()
                plt.xlim(0, n_epochs)
                val = np.array(self.errors.validation)
                train = np.array(self.errors.training)
                k_size = interval
                kernel = np.ones(k_size) / k_size
                val = np.convolve(val, kernel, mode="valid")
                train = np.convolve(train, kernel, mode="valid")

                plt.plot(train, label="Training Error")

                if validation:
                    plt.plot(val, label="Validation Error")
                plt.title("Training/Validation Errors")

                plt.xlabel("epoch")
                plt.ylabel("error")
                plt.legend()
                plt.show()
                sleep(0.3)

        return self.errors

    def predict(self, x: np.ndarray, y: np.ndarray) -> tuple[float, Any, np.ndarray]:
        return self._batch_pass(x, y)
