import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.ELU(
        alpha=(operation["args"]["alpha"] if "alpha" in operation["args"] else 1.0),
    )
