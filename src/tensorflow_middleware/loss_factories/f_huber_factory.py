import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.losses.Huber(
        delta=(operation["args"]["delta"] if "delta" in operation["args"] else 1.0),
        reduction=(operation["args"]["reduction"] if "reduction" in operation["args"] else "sum_over_batch_size"),
        name=(operation["args"]["name"] if "name" in operation["args"] else "huber"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )