__author__ = "eolus87"

# Standard libraries
import os
import datetime

# Third party libraries
import tensorflow as tf
import tf2onnx

# Custom libraries


MODELS_FOLDER = 'models'
print(tf.__version__)

model_path = os.path.join(MODELS_FOLDER, "mnist_cnn")
loaded_model = tf.keras.models.load_model(model_path)

model_proto, _ = tf2onnx.convert.from_keras(loaded_model)

output_onnx_model = os.path.join(MODELS_FOLDER, "mnist_cnn.onnx")

with open(output_onnx_model, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model saved to {output_onnx_model}")
