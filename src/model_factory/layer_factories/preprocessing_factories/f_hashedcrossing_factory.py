import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.HashedCrossing(
        num_bins=operation["args"]["num_bins"],
        output_mode=(operation["args"]["output_mode"] if "output_mode" in operation["args"] else "int"),
        sparse=(operation["args"]["sparse"] if "sparse" in operation["args"] else False),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )(self.project_data[operation["args"]["inputs"]])

    # not complete yet