import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.AveragePooling3D(
        pool_size=operation["args"]["pool_size"],
        strides=(operation["args"]["strides"] if "strides" in operation["args"] else None),
        padding=(operation["args"]["padding"] if "padding" in operation["args"] else "valid"),
        data_format=(operation["args"]["data_format"] if "data_format" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )(self.project_data[operation["args"]["inputs"]])