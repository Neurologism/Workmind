from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Normalization(
        axis=(operation["data"]["axis"] if "axis" in operation["data"] else -1),
        mean=(operation["data"]["mean"] if "mean" in operation["data"] else None),
        variance=(
            operation["data"]["variance"] if "variance" in operation["data"] else None
        ),
        invert=(
            operation["data"]["invert"] if "invert" in operation["data"] else False
        ),
    )
    self.project_data[operation["id"]].adapt(
        self.project_data[operation["data"]["dataset"][0]][
            operation["data"]["dataset"][1]
        ].map(lambda x, y: x)
    )
