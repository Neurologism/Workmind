from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.AutoContrast(
        value_range=(
            tuple(operation["args"]["value_range"])
            if "value_range" in operation["args"]
            else (0, 255)
        ),
    )
