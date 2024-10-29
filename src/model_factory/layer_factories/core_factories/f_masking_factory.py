import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Masking(
        (operation["args"]["mask_value"] if "mask_value" in operation["args"] else 0.0),
    )