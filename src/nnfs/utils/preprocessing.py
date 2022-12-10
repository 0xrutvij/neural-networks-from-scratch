from typing import Self

import numpy as np


class Preprocessing:
    def __new__(cls) -> Self:  # type: ignore
        raise Exception("this class is a namespace and cannot be instantiated.")

    @staticmethod
    def train_test_split(
        dataset: np.ndarray, test_ratio: float, shuffle: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:

        if shuffle:
            np.random.shuffle(dataset)
        n = dataset.shape[0]
        n_test = int(test_ratio * n)
        return dataset[:n_test, :], dataset[n_test:, :]

    @staticmethod
    def xy_split(
        dataset: np.ndarray, output_last: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if output_last:
            return dataset[:, :-1], dataset[:, -1:]
        else:
            return dataset[:, 1:], dataset[:, :1]

    @staticmethod
    def standard_scale(
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_mean = np.mean(data, axis=0, keepdims=True)
        d_std = np.std(data, axis=0, keepdims=True)

        data = (data - d_mean) / d_std

        return data, d_mean, d_std
