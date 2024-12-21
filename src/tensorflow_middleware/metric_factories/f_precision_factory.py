from ..m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.Precision(
        tresholds=(
            operation["args"]["tresholds"] if "tresholds" in operation["args"] else None
        ),
        top_k=(operation["args"]["top_k"] if "top_k" in operation["args"] else None),
        class_id=(
            operation["args"]["class_id"] if "class_id" in operation["args"] else None
        ),
        name=(
            operation["args"]["name"] if "name" in operation["args"] else "precision"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
