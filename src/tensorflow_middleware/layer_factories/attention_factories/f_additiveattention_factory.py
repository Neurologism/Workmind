from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.AdditiveAttention(
        use_scale=(
            operation["args"]["use_scale"] if "use_scale" in operation["args"] else True
        ),
        dropout=(
            operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0
        ),
    )
