import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GroupQueryAttention(
        head_dim=operation["args"]["head_dim"],
        num_query_heads=operation["args"]["num_query_heads"],
        num_key_value_heads=operation["args"]["num_key_value_heads"],
        dropout=(operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0),
        use_bias=(operation["args"]["use_bias"] if "use_bias" in operation["args"] else True),
        kernel_initializer=(operation["args"]["kernel_initializer"] if "kernel_initializer" in operation["args"] else "glorot_uniform"),
        bias_initializer=(operation["args"]["bias_initializer"] if "bias_initializer" in operation["args"] else "zeros"),
    )(self.project_data[operation["args"]["inputs"]])