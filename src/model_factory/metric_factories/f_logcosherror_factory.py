import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.LogCoshError(
        name=(operation["args"]["name"] if "name" in operation["args"] else "logcosh_error"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )