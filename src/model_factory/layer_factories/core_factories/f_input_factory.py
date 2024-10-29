import tensorflow as tf
import keras
from keras.src.utils.module_utils import tensorflow


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Input(
        shape=(tuple(x for x in operation["args"]["shape"]) if "shape" in operation["args"] else None),
        batch_size=(operation["args"]["batch_size"] if "batch_size" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
        sparse=(operation["args"]["sparse"] if "sparse" in operation["args"] else False),
        batch_shape=(tuple(x for x in operation["args"]["batch_shape"]) if "batch_shape" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        tensor=(self.project_data[operation["args"]["tensor"]] if "tensor" in operation["args"] else None),
        optional=(operation["args"]["optional"] if "optional" in operation["args"] else False),
    )