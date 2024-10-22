import keras

from c_whitemind_model import WhitemindProject


def call(project: WhitemindProject) -> None:
    project.keras_data[project.json_data["uid"]] = keras.layers.Input(
        shape={
            project.json_data["args"]["shape"],
        }
    )
