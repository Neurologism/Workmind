import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.CategoryEncoding(
        num_tokens=(operation["args"]["num_tokens"] if "num_tokens" in operation["args"] else None),
        output_mode=(operation["args"]["output_mode"] if "output_mode" in operation["args"] else "multi_hot"),
        sparse=(operation["args"]["sparse"] if "sparse" in operation["args"] else False),
    )(self.project_data[operation["args"]["inputs"]])

    # not complete yet