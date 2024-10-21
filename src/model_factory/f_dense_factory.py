import keras
from src.model_factory.c_whitemind_model import WhitemindProject


def call(json_data: dict, project: WhitemindProject) -> None:
    project.kerasData[json_data["uid"]] = keras.layers.Dense(
        json_data["args"]["units"],
        activation=json_data["args"]["activation"]["method"],
    )(project.kerasData[json_data["args"]["inputs"]])
