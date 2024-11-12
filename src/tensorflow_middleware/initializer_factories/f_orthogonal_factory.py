import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.Orthogonal(
        gain=(operation["args"]["gain"] if "gain" in operation["args"] else 1.0),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
