import tensorflow as tf
import keras
from keras.src.utils.module_utils import tensorflow


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.InputSpec(
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
        shape=(tuple(x for x in operation["args"]["shape"]) if "shape" in operation["args"] else None),
        ndim=(operation["args"]["ndim"] if "ndim" in operation["args"] else None),
        max_ndim=(operation["args"]["max_ndim"] if "max_ndim" in operation["args"] else None),
        min_ndim=(operation["args"]["min_ndim"] if "min_ndim" in operation["args"] else None),
        axes=(operation["args"]["axes"] if "axes" in operation["args"] else None),
        allow_last_axis_squeeze=(operation["args"]["allow_last_axis_squeeze"] if "allow_last_axis_squeeze" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        optional=(operation["args"]["optional"] if "optional" in operation["args"] else False),
    )