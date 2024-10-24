import tensorflow as tf
import keras

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Flatten(
    )(self.project_data[operation["args"]["inputs"]])