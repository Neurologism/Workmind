import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.RandomFlip(
        mode=(
            operation["args"]["mode"]
            if "mode" in operation["args"]
            else "horizontal_and_vertical"
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
    )
