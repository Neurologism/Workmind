from ..m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.BinaryAccuracy(
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "binary_accuracy"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
        threshold=(
            operation["args"]["threshold"] if "threshold" in operation["args"] else 0.5
        ),
    )
