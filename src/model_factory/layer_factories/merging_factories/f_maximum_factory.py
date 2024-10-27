import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Maximum(
    )(self.project_data[layer] for layer in operation["args"]["inputs"])