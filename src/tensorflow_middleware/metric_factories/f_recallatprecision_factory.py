import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.RecallAtPrecision(
        precision=operation["args"]["precision"],
        tresholds=(
            operation["args"]["tresholds"] if "tresholds" in operation["args"] else None
        ),
        class_id=(
            operation["args"]["class_id"] if "class_id" in operation["args"] else None
        ),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "recall_at_precision"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
