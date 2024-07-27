__author__ = "eolus87"

#%% Loading libraries
# Standard libraries
import os
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
NUM_ITERATIONS = 5
NUM_INFERENCES = 500
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

#%% Logging configuration
log_filename = os.path.join(RESULTS_FOLDER, f'{formatted_now}_tensorflow_inference_quantized.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# Optimize TensorFlow for CPU inference
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow debugging logs
# tf.config.threading.set_intra_op_parallelism_threads(4)  # Adjust based on your CPU
# tf.config.threading.set_inter_op_parallelism_threads(4)  # Adjust based on your CPU
# tf.config.set_soft_device_placement(True)  # Allow TensorFlow to automatically choose an existing and supported device

#%% Loading, conversion, saving and reloading of the model
# Load the original model
model_path = os.path.join(MODELS_FOLDER, 'mnist_cnn')
model = tf.keras.models.load_model(model_path)
logger.info(f"Loaded model from {model_path}")

# Convert the model to TFLite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model
tflite_model_path = os.path.join(MODELS_FOLDER, 'mnist_cnn_quantized.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
logger.info(f"Quantized model saved to {tflite_model_path}")

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

#%% Configuration of the model
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run inference on the quantized model
def run_inference(interpreter, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

#%% Getting the dataset
# Example inference
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%% Inferencing a single image
# Run inference on the first test sample
input_data = np.expand_dims(x_test[0], axis=[0, -1]).astype(np.float32)
start_time = time.time()
output_data = run_inference(interpreter, input_data)
end_time = time.time()

logger.info(f"Inference result: {output_data}")
logger.info(f"Inference time: {end_time - start_time} seconds")

#%% Inferencing on a loop
logger.info('Staring inference...')
time_per_iteration = []
input_data = np.expand_dims(x_test, axis=-1).astype(np.float32)
inferences_number = np.min([x_test.shape[0], NUM_INFERENCES])
for i in range(NUM_ITERATIONS):
    logger.info(f"Starting iteration {i}")
    predictions = np.zeros([inferences_number, 10])
    tic = time.time()
    for j in range(inferences_number):
        output_data = run_inference(interpreter, np.expand_dims(input_data[j], axis=0))
        predictions[j, :] = output_data[0]
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