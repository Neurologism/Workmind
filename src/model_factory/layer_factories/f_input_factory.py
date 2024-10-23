import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Input(
        shape={
            operation["args"]["shape"],
        }
    )
