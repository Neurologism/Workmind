import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.OneHotMeanIoU(
        num_classes=operation["args"]["num_classes"],
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "one_hot_mean_iou"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
        ignore_class=(
            operation["args"]["ignore_class"]
            if "ignore_class" in operation["args"]
            else None
        ),
        sparse_y_pred=(
            operation["args"]["sparse_y_pred"]
            if "sparse_y_pred" in operation["args"]
            else True
        ),
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
    )
