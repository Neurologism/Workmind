import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Conv2DTranspose(
        operation["args"]["filters"],
        operation["args"]["kernel_size"],
        strides=(operation["args"]["strides"] if "strides" in operation["args"] else (1, 1)),
        padding=(operation["args"]["padding"] if "padding" in operation["args"] else "valid"),
        data_format=(operation["args"]["data_format"] if "data_format" in operation["args"] else None),
        dilation_rate=(operation["args"]["dilation_rate"] if "dilation_rate" in operation["args"] else (1, 1)),
        activation=(operation["args"]["activation"] if "activation" in operation["args"] else None),
        use_bias=(operation["args"]["use_bias"] if "use_bias" in operation["args"] else True),
        kernel_initializer=(operation["args"]["kernel_initializer"] if "kernel_initializer" in operation["args"] else "glorot_uniform"),
        bias_initializer=(operation["args"]["bias_initializer"] if "bias_initializer" in operation["args"] else "zeros"),
        kernel_regularizer=(self.project_data[operation["args"]["kernel_regularizer"]] if "kernel_regularizer" in operation["args"] else None),
        bias_regularizer=(self.project_data[operation["args"]["bias_regularizer"]] if "bias_regularizer" in operation["args"] else None),
        activity_regularizer=(self.project_data[operation["args"]["activity_regularizer"]] if "activity_regularizer" in operation["args"] else None),
        kernel_constraint=(self.project_data[operation["args"]["kernel_constraint"]] if "kernel_constraint" in operation["args"] else None),
        bias_constraint=(self.project_data[operation["args"]["bias_constraint"]] if "bias_constraint" in operation["args"] else None),
    )