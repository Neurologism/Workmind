import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Reshape(
        target_shape=tuple(operation["data"]["target_shape"]),
    )
