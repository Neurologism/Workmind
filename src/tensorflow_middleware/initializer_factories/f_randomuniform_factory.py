from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.RandomUniform(
        minval=(
            operation["args"]["minval"] if "minval" in operation["args"] else -0.05
        ),
        maxval=(operation["args"]["maxval"] if "maxval" in operation["args"] else 0.05),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
