from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Softmax(
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
    )
