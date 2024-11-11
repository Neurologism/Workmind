import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.RNN(
        cell=self.project_data[operation["args"]["cell"]],
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
        unroll=(
            operation["args"]["unroll"] if "unroll" in operation["args"] else False
        ),
        zero_output_for_mask=(
            operation["args"]["zero_output_for_mask"]
            if "zero_output_for_mask" in operation["args"]
            else False
        ),
    )
