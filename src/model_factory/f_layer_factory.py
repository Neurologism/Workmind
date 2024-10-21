import keras
import f_dense_factory


def new(json_data: dict, keras_data: dict) -> None:
    if json_data["args"]["class"] == "Dense":
        f_dense_factory.call(json_data, keras_data)


def call(json_data: dict, keras_data: dict) -> None:
    if json_data["method"] == "new":
        new(json_data, keras_data)
