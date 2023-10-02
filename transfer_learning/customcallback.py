__author__ = "eolus87"

# Standard libraries
import time
# Third party libraries
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
# Custom libraries

plt.rcParams['figure.constrained_layout.use'] = True


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, name, epoch_number, model, x_train, x_test, y_test):
        self.name = name
        self.epoch_number = epoch_number
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.epoch_number == 0:
            y_pred = self.model.predict(self.x_test)
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
            ax[0].scatter(self.x_test, self.y_test, label='Real values', color=u'#1f77b4', zorder=1)
            ax[0].scatter(self.x_test, y_pred, label='Predictions', color=u'#ff7f0e', zorder=2)
            ax[0].axvline(np.min(self.x_train), label='Train interval_min', color='red', zorder=0)
            ax[0].axvline(np.max(self.x_train), label='Train interval_max', color='green', zorder=0)
            ax[0].set_xlabel('X values')
            ax[0].set_ylabel('Y Values')
            ax[0].set_title(f"Epoch {epoch}")
            ax[0].set_ylim(-2, 3)

            weights_array = []
            for i in range(len(self.model.get_weights())):
                weights_array.append(self.model.get_weights()[i].ravel())
            weights_array = np.concatenate(weights_array)
            ax[1].bar(np.arange(0, len(weights_array)), weights_array)
            ax[1].set_xlabel('Weights')
            ax[1].set_ylabel('Weights Value')
            ax[1].set_ylim(-2, 2)

            time.sleep(0.05)
            plt.savefig(f"pictures\\{self.name}_{epoch}.png")
            plt.close()
