from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.FBetaScore(
        average=(
            operation["args"]["average"] if "average" in operation["args"] else None
        ),
        beta=(operation["args"]["beta"] if "beta" in operation["args"] else 1.0),
        threshold=(
            operation["args"]["threshold"] if "threshold" in operation["args"] else None
        ),
        name=(
            operation["args"]["name"] if "name" in operation["args"] else "fbeta_score"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
