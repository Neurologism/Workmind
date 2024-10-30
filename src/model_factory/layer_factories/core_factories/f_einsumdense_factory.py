import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.EinsumDense(
        equation=operation["args"]["equation"],
        output_shape=(
            operation["args"]["output_shape"]
            if "output_shape" in operation["args"]
            else None
        ),
        activation=(
            operation["args"]["activation"]
            if "activation" in operation["args"]
            else None
        ),
        bias_axes=(
            operation["args"]["bias_axes"] if "bias_axes" in operation["args"] else None
        ),
        kernel_initializer=(
            operation["args"]["kernel_initializer"]
            if "kernel_initializer" in operation["args"]
            else "glorot_uniform"
        ),
        bias_initializer=(
            operation["args"]["bias_initializer"]
            if "bias_initializer" in operation["args"]
            else "zeros"
        ),
        kernel_regularizer=(
            self.project_data[operation["args"]["kernel_regularizer"]]
            if "kernel_regularizer" in operation["args"]
            else None
        ),
        bias_regularizer=(
            self.project_data[operation["args"]["bias_regularizer"]]
            if "bias_regularizer" in operation["args"]
            else None
        ),
        kernel_constraint=(
            self.project_data[operation["args"]["kernel_constraint"]]
            if "kernel_constraint" in operation["args"]
            else None
        ),
        bias_constraint=(
            self.project_data[operation["args"]["bias_constraint"]]
            if "bias_constraint" in operation["args"]
            else None
        ),
        lora_rank=(
            operation["args"]["lora_rank"] if "lora_rank" in operation["args"] else None
        ),
    )
