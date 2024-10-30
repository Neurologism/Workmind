import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GlobalAveragePooling3D(
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
        keepdims=(
            operation["args"]["keepdims"] if "keepdims" in operation["args"] else False
        ),
    )
