import tensorflow as tf
import keras
from numpy import dtype


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.losses.SparseCategoricalCrossentropy(
        from_logits=(
            operation["args"]["from_logits"]
            if "from_logits" in operation["args"]
            else False
        ),
        ignore_class=(
            operation["args"]["ignore_class"]
            if "ignore_class" in operation["args"]
            else None
        ),
        reduction=(
            operation["args"]["reduction"]
            if "reduction" in operation["args"]
            else "sum_over_batch_size"
        ),
        name=(
            operation["args"]["name"]
            if "name" in operation["args"]
            else "sparse_categorical_crossentropy"
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
