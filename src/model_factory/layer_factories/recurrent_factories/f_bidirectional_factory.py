import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Bidirectional(
        layer=self.project_data[operation["args"]["layer"]],
        merge_mode=(operation["args"]["merge_mode"] if "merge_mode" in operation["args"] else "concat"),
    )(self.project_data[operation["args"]["inputs"]])

    # not complete yet