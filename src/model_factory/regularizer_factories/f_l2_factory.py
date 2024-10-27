import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.regularizers.L2(
        l2=(operation["args"]["l2"] if "l2" in operation["args"] else 0.01),
    )