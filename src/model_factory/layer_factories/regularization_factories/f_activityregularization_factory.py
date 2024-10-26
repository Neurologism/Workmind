import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.ActivityRegularization(
        l1=operation["args"]["l1"],
        l2=operation["args"]["l2"],
    )(self.project_data[operation["args"]["inputs"]])