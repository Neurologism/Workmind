import keras


class DenseFactory:
    def dense_factory(json_data: dict, keras_data: dict) -> None:
        keras_data[json_data["uid"]] = keras.layers.Dense(
            json_data["args"]["units"],
            activation=json_data["args"]["activation"]["method"],
        )(keras_data[json_data["args"]["inputs"]])
