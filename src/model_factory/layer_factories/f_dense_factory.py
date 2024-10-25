import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Dense(
        operation["args"]["units"],
        activation=(operation["args"]["activation"] if "activation" in operation["args"] else None),
        use_bias=(operation["args"]["use_bias"] if "use_bias" in operation["args"] else True),
        kernel_initializer=(operation["args"]["kernel_initializer"] if "kernel_initializer" in operation["args"] else "glorot_uniform"),
        bias_initializer=(operation["args"]["bias_initializer"] if "bias_initializer" in operation["args"] else "zeros"),
        kernel_regularizer=(operation["args"]["kernel_regularizer"] if "kernel_regularizer" in operation["args"] else None),
        bias_regularizer=(operation["args"]["bias_regularizer"] if "bias_regularizer" in operation["args"] else None),
        activity_regularizer=(operation["args"]["activity_regularizer"] if "activity_regularizer" in operation["args"] else None),
        kernel_constraint=(operation["args"]["kernel_constraint"] if "kernel_constraint" in operation["args"] else None),
        bias_constraint=(operation["args"]["bias_constraint"] if "bias_constraint" in operation["args"] else None),
        lora_rank=(operation["args"]["lora_rank"] if "lora_rank" in operation["args"] else None),
    )(self.project_data[operation["args"]["inputs"]])
