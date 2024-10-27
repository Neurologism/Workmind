import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.RandomContrast(
        factor=operation["args"]["factor"],
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )(self.project_data[operation["args"]["inputs"]])