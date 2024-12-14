from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.CosineSimilarity(
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "cosine_similarity"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
