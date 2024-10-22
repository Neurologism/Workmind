import keras

def call(self, operation: dict) -> None:
    self.keras_data[operation["uid"]] = keras.layers.Dense(
        operation["args"]["units"],
        activation=operation["args"]["activation"]["method"],
    )(self.keras_data[operation["args"]["inputs"]])
