from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.MaxPooling3D(
        pool_size=(
            operation["args"]["pool_size"]
            if "pool_size" in operation["args"]
            else (2, 2, 2)
        ),
        strides=(
            operation["args"]["strides"] if "strides" in operation["args"] else None
        ),
        padding=(
            operation["args"]["padding"] if "padding" in operation["args"] else "valid"
        ),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
