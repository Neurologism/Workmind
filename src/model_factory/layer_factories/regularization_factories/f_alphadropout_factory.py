import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.AlphaDropout(
        rate=operation["args"]["rate"],
        noise_shape=(
            operation["args"]["noise_shape"]
            if "noise_shape" in operation["args"]
            else None
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
