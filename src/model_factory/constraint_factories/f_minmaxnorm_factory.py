import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.constraints.MinMaxNorm(
        min_value=(operation["args"]["min_value"] if "min_value" in operation["args"] else 0),
        max_value=(operation["args"]["max_value"] if "max_value" in operation["args"] else 1),
        rate=(operation["args"]["rate"] if "rate" in operation["args"] else 1),
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else 0),
    )
