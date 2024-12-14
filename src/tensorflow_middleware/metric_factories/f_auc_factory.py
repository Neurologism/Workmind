from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.AUC(
        num_tresholds=(
            operation["args"]["num_tresholds"]
            if "num_tresholds" in operation["args"]
            else 200
        ),
        curve=(operation["args"]["curve"] if "curve" in operation["args"] else "ROC"),
        summation_method=(
            operation["args"]["summation_method"]
            if "summation_method" in operation["args"]
            else "interpolation"
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
        thresholds=(
            operation["args"]["thresholds"]
            if "thresholds" in operation["args"]
            else None
        ),
        multi_label=(
            operation["args"]["multi_label"]
            if "multi_label" in operation["args"]
            else False
        ),
        num_labels=(
            operation["args"]["num_labels"]
            if "num_labels" in operation["args"]
            else None
        ),
        label_weights=(
            operation["args"]["label_weights"]
            if "label_weights" in operation["args"]
            else None
        ),
        from_logits=(
            operation["args"]["from_logits"]
            if "from_logits" in operation["args"]
            else False
        ),
    )
