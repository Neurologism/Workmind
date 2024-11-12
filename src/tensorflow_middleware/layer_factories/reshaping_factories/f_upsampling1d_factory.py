import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.UpSampling1D(
        size=operation["args"]["size"],
    )
