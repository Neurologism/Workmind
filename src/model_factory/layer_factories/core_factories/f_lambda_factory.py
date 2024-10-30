import tensorflow as tf
import keras
from keras.src.utils.module_utils import tensorflow


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Lambda(
        function=self.project_data[operation["args"]["function"]],
        output_shape=(
            tuple(x for x in operation["args"]["output_shape"])
            if "output_shape" in operation["args"]
            else None
        ),
        mask=(
            self.project_data[operation["args"]["mask"]]
            if "mask" in operation["args"]
            else None
        ),
        arguments=(
            operation["args"]["arguments"] if "arguments" in operation["args"] else None
        ),
    )
