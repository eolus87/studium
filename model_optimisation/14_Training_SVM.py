__author__ = "eolus87"

# Standard libraries
import logging
import os
import time
from datetime import datetime

# Third party libraries
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

# Custom libraries


# Script configuration and set up
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'
print(tf.__version__)

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")

#%% Logging configuration
log_filename = os.path.join(RESULTS_FOLDER, f'{formatted_now}_training_svm.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


# Loading the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the input data
logger.info("Preprocessing the input data...")
x_train = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 28*28).astype(np.float32) / 255.0    # Flatten and normalize

# Train the SVM model
logger.info("Training the SVM model...")
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(x_train[:5000,:], y_train[:5000])

# Evaluate the model
logger.info("Evaluating the model...")
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the model
model_path = os.path.join(MODELS_FOLDER, 'mnist_svm_model.joblib')
joblib.dump(svm_model, model_path)
print(f"Model saved to {model_path}")

# Test inference speed
logger.info("Measuring inference speed...")
for j in range(5):
    inference_times = []
    start_time = time.time()
    for i in range(len(x_test)):
        _ = svm_model.predict([x_test[i]])
    end_time = time.time()
    delta_time = end_time - start_time
    inference_times.append(delta_time)
    logger.info(f"Inference {j}: took {delta_time} s in total")
    logger.info(f"Inference {j}: took {delta_time/len(x_test)} s per sample")


average_inference_time = np.mean(inference_times)
print(f"Average inference time per sample: {average_inference_time * 1000:.2f} ms")