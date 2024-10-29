import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.LeakyReLU(
        negative_slope=(operation["parameters"]["alpha"] if "alpha" in operation["parameters"] else 0.3)
    )