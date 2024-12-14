from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.ConvLSTM1D(
        filters=operation["args"]["filters"],
        kernel_size=operation["args"]["kernel_size"],
        strides=(operation["args"]["strides"] if "strides" in operation["args"] else 1),
        padding=(
            operation["args"]["padding"] if "padding" in operation["args"] else "valid"
        ),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
        dilation_rate=(
            operation["args"]["dilation_rate"]
            if "dilation_rate" in operation["args"]
            else 1
        ),
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
        unit_forget_bias=(
            operation["args"]["unit_forget_bias"]
            if "unit_forget_bias" in operation["args"]
            else True
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
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        return_sequences=(
            operation["args"]["return_sequences"]
            if "return_sequences" in operation["args"]
            else False
        ),
        return_state=(
            operation["args"]["return_state"]
            if "return_state" in operation["args"]
            else False
        ),
        go_backwards=(
            operation["args"]["go_backwards"]
            if "go_backwards" in operation["args"]
            else False
        ),
        stateful=(
            operation["args"]["stateful"] if "stateful" in operation["args"] else False
        ),
    )
