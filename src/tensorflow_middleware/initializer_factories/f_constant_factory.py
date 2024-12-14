from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.Constant(
        value=(operation["args"]["value"] if "value" in operation["args"] else 0),
    )
