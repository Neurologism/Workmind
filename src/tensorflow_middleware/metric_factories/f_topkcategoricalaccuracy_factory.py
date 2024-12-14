from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.TopKCategoricalAccuracy(
        k=(operation["args"]["k"] if "k" in operation["args"] else 5),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "top_k_categorical_accuracy"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
