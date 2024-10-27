import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.regularizers.L1(
        l1=(operation["args"]["l1"] if "l1" in operation["args"] else 0.01),
    )