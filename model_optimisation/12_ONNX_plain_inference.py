__author__ = "eolus87"

# Standard libraries
import os
import logging
from datetime import datetime
import time

# Third party libraries
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Custom libraries

#%% Script configuration
NUM_ITERATIONS = 5
NUM_INFERENCES = 500
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'
print(tf.__version__)

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

#%% Configuring the logger
log_filename = os.path.join(RESULTS_FOLDER, f'{formatted_now}_onnx_inference_plain.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

#%% Loading the dataset
logger.info('Loading and preparing the dataset...')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0

#%% Loading ONNX model
logger.info("Loading the model")
session = ort.InferenceSession("models/mnist_cnn.onnx", provider=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
logger.info("Model loaded")

#%% Preprocessing the image
def preprocess_image(image: np.ndarray) -> np.ndarray:
    return np.expand_dims(image, axis=[0, -1]).astype(np.float32)

# Run single inferece
tic = time.time()
result = session.run(None, {input_name: preprocess_image(x_test[0])})
toc = time.time() - tic
logger.info(f"Inference result: {result[0]}")
logger.info(f"Inference time: {toc} seconds")

#%% Inferencing on a loop
logger.info('Starting inference...')
time_per_iteration = []
inferences_number = np.min([x_test.shape[0], NUM_INFERENCES])
for i in range(NUM_ITERATIONS):
    logger.info(f'Starting iteration {i}')
    predictions = np.zeros([inferences_number, 10])
    tic = time.time()
    for j in range(inferences_number):
        preprocess_image(x_test[j])
        prediction = session.run(None, {input_name: preprocess_image(x_test[j])})
        predictions[j, :] = prediction[0]
    toc = time.time() - tic
    predictions = np.argmax(predictions, axis=1)
    time_per_iteration.append(toc)
    logger.info(f"Inference iteration {i} took {toc} seconds")
    logger.info(f"Average inference time: {time_per_iteration[-1] / inferences_number} seconds")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test[:inferences_number], predictions)}")

average_time = np.mean(time_per_iteration)
logger.info(f"Number of inferences per iteration: {NUM_INFERENCES}")
logger.info(f"Total inference time over {NUM_ITERATIONS} iterations: {np.sum(time_per_iteration)} seconds")
logger.info(f"Average inference time over {NUM_ITERATIONS} iterations: {average_time} seconds")
logger.info(f"Standard deviation: {np.std(time_per_iteration)}")