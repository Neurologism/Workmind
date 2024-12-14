from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GroupQueryAttention(
        head_dim=operation["args"]["head_dim"],
        num_query_heads=operation["args"]["num_query_heads"],
        num_key_value_heads=operation["args"]["num_key_value_heads"],
        dropout=(
            operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0
        ),
        use_bias=(
            operation["args"]["use_bias"] if "use_bias" in operation["args"] else True
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
            operation["args"]["kernel_regularizer"]
            if "kernel_regularizer" in operation["args"]
            else None
        ),
        bias_regularizer=(
            operation["args"]["bias_regularizer"]
            if "bias_regularizer" in operation["args"]
            else None
        ),
        activity_regularizer=(
            operation["args"]["activity_regularizer"]
            if "activity_regularizer" in operation["args"]
            else None
        ),
        kernel_constraint=(
            operation["args"]["kernel_constraint"]
            if "kernel_constraint" in operation["args"]
            else None
        ),
        bias_constraint=(
            operation["args"]["bias_constraint"]
            if "bias_constraint" in operation["args"]
            else None
        ),
    )
