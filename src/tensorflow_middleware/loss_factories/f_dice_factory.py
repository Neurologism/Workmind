from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.losses.Dice(
        reduction=(
            operation["args"]["reduction"]
            if "reduction" in operation["args"]
            else "sum_over_batch_size"
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else "dice"),
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
