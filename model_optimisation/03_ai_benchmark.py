__author__ = "eolus87"

# Standard libraries
import warnings

# Third party libraries
import tensorflow as tf
from ai_benchmark import AIBenchmark
import numpy as np

# Custom libraries

np.warnings = warnings

def LSTM_Sentiment(input_):
    lstm_layer = tf.keras.layers.LSTM(1024)
    return lstm_layer(input_)

benchmark = AIBenchmark(use_CPU=True, verbose_level=1)

results = benchmark.run_inference()

print(results)