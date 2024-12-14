from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Solarization(
        addition_factor=(
            operation["args"]["addition_factor"]
            if "addition_factor" in operation["args"]
            else 0.0
        ),
        threshold_factor=(
            operation["args"]["threshold_factor"]
            if "threshold_factor" in operation["args"]
            else 0.0
        ),
        value_range=(
            tuple(operation["args"]["value_range"])
            if "value_range" in operation["args"]
            else (0, 255)
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
