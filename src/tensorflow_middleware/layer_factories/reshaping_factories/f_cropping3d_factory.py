from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Cropping3D(
        cropping=(
            tuple(operation["args"]["cropping"])
            if "cropping" in operation["args"]
            else ((1, 1), (1, 1), (1, 1))
        ),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
    )
