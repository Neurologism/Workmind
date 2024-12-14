from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.Accuracy(
        name=(operation["args"]["name"] if "name" in operation["args"] else "accuracy"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
