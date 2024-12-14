from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.SpecificityAtSensitivity(
        sensitivity=operation["args"]["sensitivity"],
        num_tresholds=(
            operation["args"]["num_trasholds"]
            if "num_trasholds" in operation["args"]
            else 200
        ),
        class_id=(
            operation["args"]["class_id"] if "class_id" in operation["args"] else None
        ),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "specificity_at_sensitivity"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
