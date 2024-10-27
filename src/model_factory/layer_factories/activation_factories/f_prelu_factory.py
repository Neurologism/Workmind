import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.PReLU(
        alpha_initializer=(operation["args"]["alpha_initializer"] if "alpha_initializer" in operation["args"] else "zeros"),
        shared_axes=(operation["args"]["shared_axes"] if "shared_axes" in operation["args"] else None),
    )(self.project_data[operation["args"]["input"]])