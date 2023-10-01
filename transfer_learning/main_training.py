__author__ = "eolus87"

# Standard libraries
# Third party libraries
import numpy as np
import tensorflow as tf
# Custom libraries
from auxiliary_functions import trigonometric, compile_and_fit_nonlinear

picture_name = "initial_trigonometric"

# Generating training and testing dataset
x_train = np.linspace(0, 2*np.pi, 256).reshape(-1, 1)
y_train = trigonometric(x_train).reshape(-1, 1)

x_test = np.linspace(-3*np.pi, 3*np.pi, 512).reshape(-1, 1)
y_test = trigonometric(x_test).reshape(-1, 1)


neurons = [6, 4, 2]
epochs = 6000
lr = 0.0015
batch_size = 256
min_val_for_random_init = -1
max_val_for_random_init = 1

model, history = compile_and_fit_nonlinear(picture_name, neurons, epochs, lr, batch_size,
                                           x_train, y_train, x_test, y_test,
                                           min_val_for_random_init, max_val_for_random_init)

model.save("models\\initial_trigonometric.keras")
