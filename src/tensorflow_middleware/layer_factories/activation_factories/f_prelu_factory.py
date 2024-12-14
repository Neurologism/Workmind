from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.PReLU(
        alpha_initializer=(
            operation["args"]["alpha_initializer"]
            if "alpha_initializer" in operation["args"]
            else "zeros"
        ),
        alpha_regularizer=(
            self.project_data[operation["args"]["alpha_regularizer"]]
            if "alpha_regularizer" in operation["args"]
            else None
        ),
        alpha_constraint=(
            self.project_data[operation["args"]["alpha_constraint"]]
            if "alpha_constraint" in operation["args"]
            else None
        ),
        shared_axes=(
            operation["args"]["shared_axes"]
            if "shared_axes" in operation["args"]
            else None
        ),
    )
