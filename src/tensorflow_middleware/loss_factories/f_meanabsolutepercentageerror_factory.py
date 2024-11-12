import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.losses.MeanAbsolutePercentageError(
        reduction=(operation["args"]["reduction"] if "reduction" in operation["args"] else "sum_over_batch_size"),
        name=(operation["args"]["name"] if "name" in operation["args"] else "mean_absolute_percentage_error"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )