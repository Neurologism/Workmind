import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Average()
