from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.Mean(
        name=(operation["args"]["name"] if "name" in operation["args"] else "mean"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
