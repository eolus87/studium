__author__ = "eolus87"

# Standard libraries
# Third party libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
# Custom libraries
from customcallback import CustomCallback


def trigonometric(x):
    y = np.sin(x)
    return y


def trigonometric_shifted(x):
    y = 2 + np.sin(x - 1)
    return y


def compile_and_fit_nonlinear(picture_name, neurons, epochs, lr, bs, x_train, y_train, x_test, y_test, minval, maxval):
    tf.keras.backend.clear_session()

    # Initialization of the model
    model = keras.Sequential()
    model.add(keras.Input(shape=(1,)))

    # Adding the layers
    for i in range(len(neurons)):
        initializer = tf.keras.initializers.RandomUniform(minval=minval,
                                                          maxval=maxval,
                                                          seed=1)
        model.add(layers.Dense(neurons[i],
                               activation="relu",
                               kernel_initializer=initializer,
                               bias_initializer=tf.keras.initializers.Ones()))
    model.add(layers.Dense(1))

    # Compiling the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(
                      learning_rate=lr
                  ),
                  metrics=['mean_squared_error', 'mean_absolute_error'])

    # Fitting the model
    history = model.fit(x_train, y_train,
                        batch_size=bs,
                        epochs=epochs,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        verbose=0,
                        callbacks=[CustomCallback(picture_name, 100, model, x_train, x_test, y_test)]
                        )

    model.summary()
    for i in range(len(model.layers)):
        print(f"Layer: {i}")
        print(model.layers[i].weights[0].numpy())
        print(model.layers[i].bias.numpy())
    return model, history
