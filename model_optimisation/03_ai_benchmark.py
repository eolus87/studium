__author__ = "eolus87"

# Standard libraries
import warnings
from datetime import datetime
import os

# Third party libraries
import tensorflow as tf
from ai_benchmark import AIBenchmark
import numpy as np
import pandas as pd

# Custom libraries

# Step 1: Configuration of the environment
# This line is required to avoid a warning message in this package
# due to a mismatch of versions of numpy and tensorflow
np.warnings = warnings

RESULTS_FOLDER = 'results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def LSTM_Sentiment(input_):
    lstm_layer = tf.keras.layers.LSTM(1024)
    return lstm_layer(input_)

# Step 2: Initialization and running of the benchmark
benchmark = AIBenchmark(use_CPU=True, verbose_level=1)
results = benchmark.run_inference()

# Step 3: Save the DataFrame to a file
# Formatting the file name
now = datetime.now()
formatted_now = now.strftime("%y%m%d_%H%M%S")
filename = f'{formatted_now}_AI_benchmark_inference_result.txt'
filepath = os.path.join(RESULTS_FOLDER, filename)

# For a strange reason the run inference method change the cwd
homepath = os.path.dirname(os.path.abspath(__file__))
os.chdir(homepath)
# Storing the data
with open(filepath, 'w') as f:
    f.write(f'Result of inference benchmark: {results.inference_score}')