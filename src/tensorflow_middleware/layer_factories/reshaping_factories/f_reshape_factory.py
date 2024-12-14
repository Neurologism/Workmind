from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Reshape(
        target_shape=tuple(operation["data"]["target_shape"]),
    )
