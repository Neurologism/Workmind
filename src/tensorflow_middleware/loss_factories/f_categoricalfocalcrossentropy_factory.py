from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.losses.CategoricalFocalCrossentropy(
        alpha=(operation["args"]["alpha"] if "alpha" in operation["args"] else 0.25),
        gamma=(operation["args"]["gamma"] if "gamma" in operation["args"] else 2.0),
        from_logits=(
            operation["args"]["from_logits"]
            if "from_logits" in operation["args"]
            else False
        ),
        label_smoothing=(
            operation["args"]["label_smoothing"]
            if "label_smoothing" in operation["args"]
            else 0
        ),
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
        reduction=(
            operation["args"]["reduction"]
            if "reduction" in operation["args"]
            else "sum_over_batch_size"
        ),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "categorical_focal_crossentropy"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
