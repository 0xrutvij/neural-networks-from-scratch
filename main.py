import numpy as np
import pandas as pd

from nnfs.activations import Activation
from nnfs.layers import ActivationLayer, Dense, Dropout
from nnfs.losses import LossFunction
from nnfs.models.neural_network import NerualNetwork
from nnfs.optimizers import Optimizer
from nnfs.utils import Preprocessing

np.set_printoptions(precision=4)

# configuration
TEST_RATIO = 0.15
VALIDATION_RATIO = 0.15


np.random.seed(42)
white_wine_csv = "data/raw/winequality-white.csv"
red_wine_csv = "data/raw/winequality-red.csv"

white_wine = pd.read_csv(white_wine_csv, delimiter=";")

# display(white_wine)
# display(white_wine.describe())
white_wine_raw = white_wine.to_numpy()

red_wine = pd.read_csv(red_wine_csv, delimiter=";")
red_wine_raw = red_wine.to_numpy()

wines_raw = np.concatenate(white_wine_raw)
# red_wine_raw

wines_raw, wines_mean, wines_std = Preprocessing.standard_scale(wines_raw)

test, train = Preprocessing.train_test_split(
    white_wine_raw, TEST_RATIO + VALIDATION_RATIO, shuffle=True
)
test, validation = Preprocessing.train_test_split(
    test, (TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO))
)

x_train, y_train = Preprocessing.xy_split(train)
x_test, y_test = Preprocessing.xy_split(test)
x_val, y_val = Preprocessing.xy_split(validation)


# Counter(wines_raw[:, -1].tolist())

# print(x_train.shape, x_test.shape, x_val.shape)


batch_size = 10
optim = Optimizer.adam(learning_rate=0.01)

model = NerualNetwork(
    optimizer=optim,
    loss_fn=LossFunction.squared_error(),
    validation_data=(x_val, y_val),
)


prev_shape = model.add(Dense(22, (batch_size, x_train.shape[1]), optim))

prev_shape = model.add(ActivationLayer(prev_shape, Activation.relu()))

prev_shape = model.add(Dense(44, prev_shape, optim))

prev_shape = model.add(ActivationLayer(prev_shape, Activation.relu()))

prev_shape = model.add(Dropout(prev_shape, p_drop=0.25))

prev_shape = model.add(Dense(22, prev_shape, optim))

prev_shape = model.add(ActivationLayer(prev_shape, Activation.relu()))

prev_shape = model.add(Dropout(prev_shape, p_drop=0.25))

output_shape = model.add(Dense(1, prev_shape, optim))

# print(output_shape)


# print(x_train.shape, y_train.shape)


errs = model.fit(x_train, y_train, 10, batch_size=batch_size)

print(errs)
