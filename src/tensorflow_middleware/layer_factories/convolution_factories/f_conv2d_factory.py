from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.Conv2D(
        operation["data"]["filters"],
        operation["data"]["kernel_size"],
        strides=(
            operation["data"]["strides"] if "strides" in operation["data"] else (1, 1)
        ),
        padding=(
            operation["data"]["padding"] if "padding" in operation["data"] else "valid"
        ),
        data_format=(
            operation["data"]["data_format"]
            if "data_format" in operation["data"]
            else None
        ),
        dilation_rate=(
            operation["data"]["dilation_rate"]
            if "dilation_rate" in operation["data"]
            else (1, 1)
        ),
        groups=(operation["data"]["groups"] if "groups" in operation["data"] else 1),
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
    )
