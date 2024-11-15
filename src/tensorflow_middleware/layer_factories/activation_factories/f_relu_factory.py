import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.ReLU(
        max_value=(operation["data"]["max_value"] if "max_value" in operation["data"] else None),
        negative_slope=(operation["data"]["negative_slope"] if "negative_slope" in operation["data"] else 0),
        threshold=(operation["data"]["threshold"] if "threshold" in operation["data"] else 0),
    )
