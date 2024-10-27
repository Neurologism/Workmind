import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Embedding(
        operation["args"]["input_dim"],
        operation["args"]["output_dim"],
        embeddings_initializer=(operation["args"]["embeddings_initializer"] if "embeddings_initializer" in operation["args"] else "uniform"),
        mask_zero=(operation["args"]["mask_zero"] if "mask_zero" in operation["args"] else False),
        lora_rank=(operation["args"]["lora_rank"] if "lora_rank" in operation["args"] else None)
    )(self.project_data[operation["args"]["inputs"]])

# not complete
