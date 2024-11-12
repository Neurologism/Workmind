import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.Constant(
        value=(operation["args"]["value"] if "value" in operation["args"] else 0),
    )
