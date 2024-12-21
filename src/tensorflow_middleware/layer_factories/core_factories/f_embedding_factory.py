from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Embedding(
        operation["data"]["input_dim"],
        operation["data"]["output_dim"],
        embeddings_initializer=(
            operation["data"]["embeddings_initializer"]
            if "embeddings_initializer" in operation["data"]
            else "uniform"
        ),
        embeddings_regularizer=(
            self.project_data[operation["data"]["embeddings_regularizer"]]
            if "embeddings_regularizer" in operation["data"]
            else None
        ),
        embeddings_constraint=(
            self.project_data[operation["data"]["embeddings_constraint"]]
            if "embeddings_constraint" in operation["data"]
            else None
        ),
        mask_zero=(
            operation["data"]["mask_zero"]
            if "mask_zero" in operation["data"]
            else False
        ),
        weights=(
            self.project_data[operation["data"]["weights"]]
            if "weights" in operation["data"]
            else None
        ),
        lora_rank=(
            operation["data"]["lora_rank"] if "lora_rank" in operation["data"] else None
        ),
    )
