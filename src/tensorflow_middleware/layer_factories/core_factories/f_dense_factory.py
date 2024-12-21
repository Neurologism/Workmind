from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Dense(
        units=operation["data"]["units"],
        activation=(
            operation["data"]["activation"]
            if "activation" in operation["data"]
            else None
        ),
        use_bias=(
            operation["data"]["use_bias"] if "use_bias" in operation["data"] else True
        ),
        kernel_initializer=(
            operation["data"]["kernel_initializer"]
            if "kernel_initializer" in operation["data"]
            else "glorot_uniform"
        ),
        bias_initializer=(
            operation["data"]["bias_initializer"]
            if "bias_initializer" in operation["data"]
            else "zeros"
        ),
        kernel_regularizer=(
            self.project_data[operation["data"]["kernel_regularizer"]]
            if "kernel_regularizer" in operation["data"]
            else None
        ),
        bias_regularizer=(
            self.project_data[operation["data"]["bias_regularizer"]]
            if "bias_regularizer" in operation["data"]
            else None
        ),
        activity_regularizer=(
            self.project_data[operation["data"]["activity_regularizer"]]
            if "activity_regularizer" in operation["data"]
            else None
        ),
        kernel_constraint=(
            self.project_data[operation["data"]["kernel_constraint"]]
            if "kernel_constraint" in operation["data"]
            else None
        ),
        bias_constraint=(
            self.project_data[operation["data"]["bias_constraint"]]
            if "bias_constraint" in operation["data"]
            else None
        ),
        lora_rank=(
            operation["data"]["lora_rank"] if "lora_rank" in operation["data"] else None
        ),
    )
