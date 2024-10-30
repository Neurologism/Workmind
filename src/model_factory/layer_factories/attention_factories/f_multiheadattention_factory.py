import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.MultiHeadAttention(
        num_heads=operation["args"]["num_heads"],
        key_dim=operation["args"]["key_dim"],
        value_dim=(
            operation["args"]["value_dim"] if "value_dim" in operation["args"] else None
        ),
        dropout=(
            operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0
        ),
        use_bias=(
            operation["args"]["use_bias"] if "use_bias" in operation["args"] else True
        ),
        output_shape=(
            operation["args"]["output_shape"]
            if "output_shape" in operation["args"]
            else None
        ),
        attention_axes=(
            operation["args"]["attention_axes"]
            if "attention_axes" in operation["args"]
            else None
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
        activity_regularizer=(
            self.project_data[operation["args"]["activity_regularizer"]]
            if "activity_regularizer" in operation["args"]
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
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
