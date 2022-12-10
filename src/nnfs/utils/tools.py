from typing import Iterable, Optional, Self

import numpy as np


class Tools:
    """internal utility functions"""

    def __new__(cls) -> Self:  # type: ignore
        raise Exception("this class is a namespace and cannot be instantiated.")

    @staticmethod
    def batch_generator(
        x: np.ndarray, y: Optional[np.ndarray], batch_size: int = 16
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        n = x.shape[0]
        for i in range(0, n, batch_size):
            begin, end = i, min(i + batch_size, n)
            if end - begin < batch_size:
                break
            bx = by = x[begin:end]
            if y is not None:
                by = y[begin:end]
            yield bx, by
