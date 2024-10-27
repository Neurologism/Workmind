import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.regularizers.OrthogonalRegularizer(
        factor=(operation["args"]["factor"] if "factor" in operation["args"] else 0.01),
        mode=(operation["args"]["mode"] if "mode" in operation["args"] else "rows"),
    )