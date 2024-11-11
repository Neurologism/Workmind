import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.SpatialDropout3D(
        rate=operation["args"]["rate"],
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
