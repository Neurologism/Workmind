from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Pipeline(
        layers=(self.project_data[layer] for layer in operation["args"]["layers"]),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
