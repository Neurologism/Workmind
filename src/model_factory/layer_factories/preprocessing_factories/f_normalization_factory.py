import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Normalization(
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
        mean=(operation["args"]["mean"] if "mean" in operation["args"] else None),
        variance=(operation["args"]["variance"] if "variance" in operation["args"] else None),
        invert=(operation["args"]["invert"] if "invert" in operation["args"] else False),
    )