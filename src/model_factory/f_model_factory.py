import keras


def call(json_data: dict, keras_data: dict) -> None:
    if json_data["method"] == "new":
        keras_data[json_data["uid"]] = keras.Model(
            json_data["args"]["inputs"], json_data["args"]["outputs"]
        )

    elif json_data["method"] == "compile":
        keras_data[json_data["uid"]].compile(
            optimizer=json_data["args"]["optimizer"]["method"],
            loss=json_data["args"]["loss"][0]["method"],
            metrics=json_data["args"]["metrics"][0]["method"],
        )

    elif json_data["method"] == "fit":
        keras_data[json_data["uid"]].fit(
            intputs=keras_data[json_data["args"]["inputs"]],
            epochs=json_data["args"]["epochs"],
            batch_size=json_data["args"]["batch_size"],
        )
