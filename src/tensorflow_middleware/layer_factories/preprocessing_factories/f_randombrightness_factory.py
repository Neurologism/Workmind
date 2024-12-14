from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.RandomBrightness(
        factor=operation["args"]["factor"],
        value_range=(
            tuple(operation["args"]["value_range"])
            if "value_range" in operation["args"]
            else (0, 255)
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
