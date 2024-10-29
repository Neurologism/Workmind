import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Cropping1D(
        cropping=(tuple(operation["args"]["cropping"]) if "cropping" in operation["args"] else (1, 1)),
    )