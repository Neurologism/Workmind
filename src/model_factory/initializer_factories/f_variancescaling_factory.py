import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.VarianceScaling(
        scale=(operation["args"]["scale"] if "scale" in operation["args"] else 1.0),
        mode=(operation["args"]["mode"] if "mode" in operation["args"] else "fan_in"),
        distribution=(
            operation["args"]["distribution"]
            if "distribution" in operation["args"]
            else "truncated_normal"
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
