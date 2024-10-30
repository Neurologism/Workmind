import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.ReLU(
        max_value=operation["args"]["max_value"],
        negative_slope=operation["args"]["negative_slope"],
        threshold=operation["args"]["threshold"],
    )
