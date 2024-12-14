from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.Poisson(
        name=(operation["args"]["name"] if "name" in operation["args"] else "poisson"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
