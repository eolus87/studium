__author__ = "eolus87"

#%% Loading libraries
# Standard libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"]= "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= "8"
os.environ["TF_NUM_INTRAOP_THREADS"]= "8"
os.environ["TF_NUM_INTEROP_THREADS"]= "8"
import logging
import time
from datetime import datetime

# Third party libraries
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# Custom libraries

print(tf.__version__)

#%% Script configuration
# Script configuration and set up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
tf.get_logger().setLevel(logging.ERROR)  # Suppress progress bars and other verbose output
NUM_ITERATIONS = 5
NUM_INFERENCES = 500
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

#%% Logging configuration
log_filename = os.path.join(RESULTS_FOLDER, f'{formatted_now}_tensorflow_inference_optimised_loop.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

#%% Loading the model
model_folder = os.path.join(MODELS_FOLDER, 'mnist_cnn')
logger.info('Loading the model...')
loaded_model = tf.keras.models.load_model(model_folder)
logger.info("Model loaded")

#%% Getting the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%% Inferencing a single image
start_time = time.time()
predictions = loaded_model.predict(np.expand_dims(x_test[0], axis=0), verbose=0)
end_time = time.time()

logger.info(f"Inference result: {predictions}")
logger.info(f"Inference time: {end_time - start_time} seconds")

#%% Inferencing on a loop
logger.info('Starting inference...')
time_per_iteration = []
inferences_number = np.min([x_test.shape[0], NUM_INFERENCES])
for i in range(NUM_ITERATIONS):
    logger.info(f'Starting iteration {i}')
    predictions = np.zeros([inferences_number, 10])
    tic = time.time()
    for j in range(inferences_number):
        prediction = loaded_model.predict(np.expand_dims(x_test[j], axis=0), verbose=0)
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