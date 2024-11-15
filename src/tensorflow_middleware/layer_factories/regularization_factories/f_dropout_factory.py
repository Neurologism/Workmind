import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Dropout(
        rate=operation["data"]["rate"],
        noise_shape=(
            operation["data"]["noise_shape"]
            if "noise_shape" in operation["data"]
            else None
        ),
        seed=(operation["data"]["seed"] if "seed" in operation["data"] else None),
    )
