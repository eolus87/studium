__author__ = "eolus87"

# Standard libraries
# Third party libraries
import numpy as np
import tensorflow as tf
# Custom libraries
from auxiliary_functions import load_and_fit_nonlinear


def trigonometric_shifted(x):
    y = 1 + np.sin(x - 1)
    return y


picture_name = "transfer_trigonometric"
model_name = "initial_trigonometric.keras"

# Generating training and testing dataset
x_train = np.linspace(0, 2*np.pi, 24).reshape(-1, 1)
y_train = trigonometric_shifted(x_train).reshape(-1, 1)

x_test = np.linspace(-3*np.pi, 3*np.pi, 512).reshape(-1, 1)
y_test = trigonometric_shifted(x_test).reshape(-1, 1)

epochs = 6000
lr = 0.0001
batch_size = 256

model, history = load_and_fit_nonlinear(picture_name, model_name, epochs, lr, batch_size,
                                        x_train, y_train, x_test, y_test)

model.save("models\\transferred_trigonometric.keras")
