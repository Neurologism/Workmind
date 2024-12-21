from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Flatten(
        data_format=(
            operation["data"]["data_format"]
            if "data_format" in operation["data"]
            else None
        ),
    )
