from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.losses.Tversky(
        alpha=(operation["args"]["alpha"] if "alpha" in operation["args"] else 0.3),
        beta=(operation["args"]["beta"] if "beta" in operation["args"] else 0.7),
        reduction=(
            operation["args"]["reduction"]
            if "reduction" in operation["args"]
            else "sum_over_batch_size"
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else "tversky"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
