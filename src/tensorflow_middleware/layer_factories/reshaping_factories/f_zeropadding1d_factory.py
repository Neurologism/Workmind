from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.ZeroPadding1D(
        padding=operation["args"]["padding"],
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
    )
