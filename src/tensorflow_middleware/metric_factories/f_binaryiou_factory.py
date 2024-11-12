import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.BinaryIoU(
        target_class_ids=(
            operation["args"]["target_class_ids"]
            if "target_class_ids" in operation["args"]
            else (0, 1)
        ),
        treshold=(
            operation["args"]["tresholds"] if "tresholds" in operation["args"] else 0.5
        ),
        name=(
            operation["args"]["name"] if "name" in operation["args"] else "binary_iou"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
