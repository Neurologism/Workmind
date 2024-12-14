from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.RootMeanSquaredError(
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "root_mean_squared_error"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
