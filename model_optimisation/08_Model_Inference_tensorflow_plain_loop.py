__author__ = "eolus87"

# Standard libraries
import os
import logging
import time
from datetime import datetime

# Third party libraries
import tensorflow as tf
import numpy as np

# Custom libraries


# Script configuration and set up
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
tf.get_logger().setLevel(logging.ERROR)  # Suppress progress bars and other verbose output
NUM_ITERATIONS = 5
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'
print(tf.__version__)

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

log_filename = os.path.join(RESULTS_FOLDER, f'{formatted_now}_tensorflow_inference_plain.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# Loading the model
model_folder = os.path.join(MODELS_FOLDER, 'mnist_cnn')
logger.info('Loading the model...')
loaded_model = tf.keras.models.load_model(model_folder)
logger.info("Model loaded")

# Getting the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Inferencing on the model
logger.info('Starting inference...')
inferencing_times = []
for i in range(NUM_ITERATIONS):
    logger.info(f'Starting iteration {i}')
    tic = time.time()
    for j in range(x_train.shape[0]):
        predictions = loaded_model.predict(np.expand_dims(x_train[j], axis=0), verbose=0)
    toc = time.time() - tic
    inferencing_times.append(toc)
    logger.info(f'Inference iteration {i} took {toc} seconds')

logger.info(f'Average inference time: {sum(inferencing_times) / len(inferencing_times)}')
logger.info(f"Std deviation: {sum([(x - sum(inferencing_times) / len(inferencing_times)) ** 2 for x in inferencing_times]) / len(inferencing_times)}")