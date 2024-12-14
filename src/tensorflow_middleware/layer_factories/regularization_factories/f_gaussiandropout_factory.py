from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GaussianDropout(
        rate=operation["args"]["rate"],
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
