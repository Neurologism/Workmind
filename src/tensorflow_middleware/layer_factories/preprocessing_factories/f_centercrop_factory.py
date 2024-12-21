from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.CenterCrop(
        height=(operation["args"]["height"] if "height" in operation["args"] else None),
        width=(operation["args"]["width"] if "width" in operation["args"] else None),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
    )
