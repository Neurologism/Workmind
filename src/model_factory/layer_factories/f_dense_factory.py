import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Dense(
        operation["args"]["units"],
        activation=operation["args"]["activation"]["method"],
    )(self.project_data[operation["args"]["inputs"]])
