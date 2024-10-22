import keras

from c_whitemind_model import WhitemindProject


def call(project: WhitemindProject) -> None:
    project.kerasData[project.json_data["uid"]] = keras.layers.Dense(
        project.json_data["args"]["units"],
        activation=project.json_data["args"]["activation"]["method"],
    )(project.kerasData[project.json_data["args"]["inputs"]])
