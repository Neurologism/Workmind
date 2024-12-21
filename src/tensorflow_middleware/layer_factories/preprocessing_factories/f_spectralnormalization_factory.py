from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.SpectralNormalization(
        layer=self.project_data[operation["args"]["layer"]],
        power_iterations=(
            operation["args"]["power_iterations"]
            if "power_iterations" in operation["args"]
            else 1
        ),
    )
