from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.AveragePooling2D(
        pool_size=operation["data"]["pool_size"],
        strides=(
            operation["data"]["strides"] if "strides" in operation["data"] else None
        ),
        padding=(
            operation["data"]["padding"] if "padding" in operation["data"] else "valid"
        ),
        data_format=(
            operation["data"]["data_format"]
            if "data_format" in operation["data"]
            else None
        ),
        name=(operation["data"]["name"] if "name" in operation["data"] else None),
    )
