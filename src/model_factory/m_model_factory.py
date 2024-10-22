import keras

def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        self.keras_data[operation["uid"]] = keras.Model(
            self.keras_data[operation["args"]["inputs"]],
            self.keras_data[operation["args"]["outputs"]]
        )

    elif operation["method"] == "compile":
        self.keras_data[operation["uid"]].compile(
            optimizer=operation["args"]["optimizer"]["method"],
            loss=operation["args"]["loss"][0]["method"],
            metrics=[operation["args"]["metrics"][0]["method"]],
        )

    elif operation["method"] == "fit":
        self.keras_data[operation["uid"]].fit(
            intputs=self.keras_data[operation["args"]["inputs"]],
            epochs=operation["args"]["epochs"],
            batch_size=operation["args"]["batch_size"],
        )
