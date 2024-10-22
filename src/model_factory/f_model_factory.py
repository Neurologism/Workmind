import keras

def call(self) -> None:
    if self.json_data["method"] == "new":
        self.keras_data[self.json_data["uid"]] = keras.Model(
            self.keras_data[self.json_data["args"]["inputs"]],
            self.keras_data[self.json_data["args"]["outputs"]],
        )

    elif self.json_data["method"] == "compile":
        self.keras_data[self.json_data["uid"]].compile(
            optimizer=self.json_data["args"]["optimizer"]["method"],
            loss=self.json_data["args"]["loss"][0]["method"],
            metrics=self.json_data["args"]["metrics"][0]["method"],
        )

    elif self.json_data["method"] == "fit":
        self.keras_data[self.json_data["uid"]].fit(
            intputs=self.keras_data[self.json_data["args"]["inputs"]],
            epochs=self.json_data["args"]["epochs"],
            batch_size=self.json_data["args"]["batch_size"],
        )
