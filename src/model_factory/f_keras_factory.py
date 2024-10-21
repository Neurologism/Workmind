import keras
import f_layer_factory

keras_data = {}


def call(json_data: dict) -> None:
    for i in json_data["operations"]:
        datatype = i["type"]
        if datatype == "layer":
            f_layer_factory.call(i, keras_data)
