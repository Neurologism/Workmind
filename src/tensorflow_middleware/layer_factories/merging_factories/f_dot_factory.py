from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Dot(
        axes=operation["args"]["axes"],
        normalize=(
            operation["args"]["normalize"]
            if "normalize" in operation["args"]
            else False
        ),
    )
