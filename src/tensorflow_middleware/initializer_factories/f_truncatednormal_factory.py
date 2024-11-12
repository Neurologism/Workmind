import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.TruncatedNormal(
        mean=(operation["args"]["mean"] if "mean" in operation["args"] else 0.0),
        stddev=(operation["args"]["stddev"] if "stddev" in operation["args"] else 0.05),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
