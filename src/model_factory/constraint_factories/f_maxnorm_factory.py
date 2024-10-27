import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.constraints.MaxNorm(
        max_value=(operation["args"]["max_value"] if "max_value" in operation["args"] else 2),
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else 0),
    )