import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Hashing(
        num_bins=operation["args"]["num_bins"],
        mask_value=(operation["args"]["mask_value"] if "mask_value" in operation["args"] else None),
        salt=(operation["args"]["salt"] if "salt" in operation["args"] else None),
        output_mode=(operation["args"]["output_mode"] if "output_mode" in operation["args"] else "int"),
        sparse=(operation["args"]["sparse"] if "sparse" in operation["args"] else False),
    )