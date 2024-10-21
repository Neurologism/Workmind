from c_whitemind_model import WhitemindProject
import keras


def call(json_data: dict, project: WhitemindProject) -> None:
    project.kerasData[json_data["uid"]] = keras.layers.Input(
        shape={
            json_data["args"]["shape"],
        }
    )
