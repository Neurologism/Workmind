from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Rescaling(
        scale=(operation["args"]["scale"] if "scale" in operation["args"] else 1.0),
        offset=(operation["args"]["offset"] if "offset" in operation["args"] else 0.0),
    )
