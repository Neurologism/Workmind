from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Cropping2D(
        cropping=(
            tuple(operation["args"]["cropping"])
            if "cropping" in operation["args"]
            else ((0, 0), (0, 0))
        ),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
    )
