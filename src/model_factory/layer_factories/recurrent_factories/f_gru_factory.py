import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GRU(
        units=operation["args"]["units"],
        activation=(operation["args"]["activation"] if "activation" in operation["args"] else "tanh"),
        recurrent_activation=(operation["args"]["recurrent_activation"] if "recurrent_activation" in operation["args"] else "hard_sigmoid"),
        use_bias=(operation["args"]["use_bias"] if "use_bias" in operation["args"] else True),
        kernel_initializer=(operation["args"]["kernel_initializer"] if "kernel_initializer" in operation["args"] else "glorot_uniform"),
        recurrent_initializer=(operation["args"]["recurrent_initializer"] if "recurrent_initializer" in operation["args"] else "orthogonal"),
        bias_initializer=(operation["args"]["bias_initializer"] if "bias_initializer" in operation["args"] else "zeros"),
        kernel_regularizer=(self.project_data[operation["args"]["kernel_regularizer"]] if "kernel_regularizer" in operation["args"] else None),
        recurrent_regularizer=(self.project_data[operation["args"]["recurrent_regularizer"]] if "recurrent_regularizer" in operation["args"] else None),
        bias_regularizer=(self.project_data[operation["args"]["bias_regularizer"]] if "bias_regularizer" in operation["args"] else None),
        activity_regularizer=(self.project_data[operation["args"]["activity_regularizer"]] if "activity_regularizer" in operation["args"] else None),
        kernel_constraint=(self.project_data[operation["args"]["kernel_constraint"]] if "kernel_constraint" in operation["args"] else None),
        recurrent_constraint=(self.project_data[operation["args"]["recurrent_constraint"]] if "recurrent_constraint" in operation["args"] else None),
        bias_constraint=(self.project_data[operation["args"]["bias_constraint"]] if "bias_constraint" in operation["args"] else None),
        dropout=(operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0),
        recurrent_dropout=(operation["args"]["recurrent_dropout"] if "recurrent_dropout" in operation["args"] else 0.0),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        return_sequences=(operation["args"]["return_sequences"] if "return_sequences" in operation["args"] else False),
        return_state=(operation["args"]["return_state"] if "return_state" in operation["args"] else False),
        go_backwards=(operation["args"]["go_backwards"] if "go_backwards" in operation["args"] else False),
        stateful=(operation["args"]["stateful"] if "stateful" in operation["args"] else False),
        unroll=(operation["args"]["unroll"] if "unroll" in operation["args"] else False),
        reset_after=(operation["args"]["reset_after"] if "reset_after" in operation["args"] else False),
        use_cudnn=(operation["args"]["use_cudnn"] if "use_cudnn" in operation["args"] else "auto"),
    )