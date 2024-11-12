import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.TrueNegatives(
        tresholds=(
            operation["args"]["tresholds"] if "tresholds" in operation["args"] else None
        ),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "true_negatives"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
