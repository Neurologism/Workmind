from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.F1Score(
        average=(
            operation["args"]["average"] if "average" in operation["args"] else None
        ),
        threshold=(
            operation["args"]["threshold"] if "threshold" in operation["args"] else None
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else "f1_score"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
