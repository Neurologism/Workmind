import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GaussianNoise(
        stddev=operation["args"]["stddev"],
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )(self.project_data[operation["args"]["inputs"]])