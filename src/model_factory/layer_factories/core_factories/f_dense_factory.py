import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Dense(
        operation["args"]["units"],
        activation=(operation["args"]["activation"] if "activation" in operation["args"] else None),
        use_bias=(operation["args"]["use_bias"] if "use_bias" in operation["args"] else True),
        kernel_initializer=(operation["args"]["kernel_initializer"] if "kernel_initializer" in operation["args"] else "glorot_uniform"),
        bias_initializer=(operation["args"]["bias_initializer"] if "bias_initializer" in operation["args"] else "zeros"),
        lora_rank=(operation["args"]["lora_rank"] if "lora_rank" in operation["args"] else None),
    )(self.project_data[operation["args"]["inputs"]])

# not complete