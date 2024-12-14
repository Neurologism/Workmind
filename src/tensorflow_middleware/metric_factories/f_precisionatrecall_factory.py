from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.PrecisionAtRecall(
        recall=operation["args"]["recall"],
        num_tresholds=(
            operation["args"]["num_tresholds"]
            if "num_tresholds" in operation["args"]
            else 200
        ),
        class_id=(
            operation["args"]["class_id"] if "class_id" in operation["args"] else None
        ),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "precision_at_recall"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
