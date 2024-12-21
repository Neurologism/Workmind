from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Bidirectional(
        layer=self.project_data[operation["args"]["layer"]],
        merge_mode=(
            operation["args"]["merge_mode"]
            if "merge_mode" in operation["args"]
            else "concat"
        ),
        weights=(
            self.project_data[operation["args"]["weights"]]
            if "weights" in operation["args"]
            else None
        ),
        backward_layer=(
            self.project_data[operation["args"]["backward_layer"]]
            if "backward_layer" in operation["args"]
            else None
        ),
    )
