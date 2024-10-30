import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GRUCell(
        units=operation["args"]["units"],
        activation=(
            operation["args"]["activation"]
            if "activation" in operation["args"]
            else "tanh"
        ),
        recurrent_activation=(
            operation["args"]["recurrent_activation"]
            if "recurrent_activation" in operation["args"]
            else "hard_sigmoid"
        ),
        use_bias=(
            operation["args"]["use_bias"] if "use_bias" in operation["args"] else True
        ),
        kernel_initializer=(
            operation["args"]["kernel_initializer"]
            if "kernel_initializer" in operation["args"]
            else "glorot_uniform"
        ),
        recurrent_initializer=(
            operation["args"]["recurrent_initializer"]
            if "recurrent_initializer" in operation["args"]
            else "orthogonal"
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
        recurrent_regularizer=(
            self.project_data[operation["args"]["recurrent_regularizer"]]
            if "recurrent_regularizer" in operation["args"]
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
        recurrent_constraint=(
            self.project_data[operation["args"]["recurrent_constraint"]]
            if "recurrent_constraint" in operation["args"]
            else None
        ),
        bias_constraint=(
            self.project_data[operation["args"]["bias_constraint"]]
            if "bias_constraint" in operation["args"]
            else None
        ),
        dropout=(
            operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0
        ),
        recurrent_dropout=(
            operation["args"]["recurrent_dropout"]
            if "recurrent_dropout" in operation["args"]
            else 0.0
        ),
        reset_after=(
            operation["args"]["reset_after"]
            if "reset_after" in operation["args"]
            else True
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
