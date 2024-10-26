import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Subtract(
    )(self.project_data[layer] for layer in operation["args"]["inputs"])