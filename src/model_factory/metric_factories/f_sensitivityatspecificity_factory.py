import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.SensitivityAtSpecificity(
        specificity=operation["args"]["specificity"],
        num_thresholds=(operation["args"]["num_thresholds"] if "num_thresholds" in operation["args"] else 200),
        class_id=(operation["args"]["class_id"] if "class_id" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else "sensitivity_at_specificity"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )