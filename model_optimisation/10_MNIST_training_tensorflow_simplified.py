__author__ = "eolus87"
# The following code has been copied from 
# https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial
# All the credits to its author. I have just modified it slightly to 
# adatp to my code style, added some comments and some extra options.

# Standard libraries
import time
import os
from datetime import datetime
import logging

# Third party libraries
import tensorflow as tf
import numpy as np

# Custom libraries


# Script configuration and set up
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'
print(tf.__version__)

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

log_filename = os.path.join(RESULTS_FOLDER, f'{formatted_now}_training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# Getting the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the data
input_shape = (28, 28, 1)
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0

# Working with the labels
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# Model paramters
batch_size = 64
num_classes = 10
epochs = 5

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (5,5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(16, (5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

# Training the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])

logger.debug(model.summary())

# Evaluate the model
tic = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test)
toc = time.time()-tic
logger.info(f"Model evaluation performed in: {toc} seconds")
logger.info(f"Test loss: {test_loss}")
logger.info(f"Test accuracy: {test_acc}")

tic = time.time()
model.predict(x_train)
toc = time.time()-tic
logger.info(f"Model inference performed in x_train in: {toc} seconds")

# Save the model
model.save(os.path.join(MODELS_FOLDER,'mnist_cnn_simplified'))