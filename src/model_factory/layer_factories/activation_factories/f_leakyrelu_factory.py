import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.LeakyReLU(
        negative_slope=operation["args"]["negative_slope"],
    )(self.project_data[operation["args"]["input"]])