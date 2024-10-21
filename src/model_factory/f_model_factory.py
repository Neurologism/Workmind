import keras

from src.model_factory.c_whitemind_model import WhitemindProject


def call(json_data: dict, project: WhitemindProject) -> None:
    if json_data["method"] == "new":
        project.kerasData[json_data["uid"]] = keras.Model(
            project.kerasData[json_data["args"]["inputs"]],
            project.kerasData[json_data["args"]["outputs"]],
        )

    elif json_data["method"] == "compile":
        project.kerasData[json_data["uid"]].compile(
            optimizer=json_data["args"]["optimizer"]["method"],
            loss=json_data["args"]["loss"][0]["method"],
            metrics=json_data["args"]["metrics"][0]["method"],
        )

    elif json_data["method"] == "fit":
        project.kerasData[json_data["uid"]].fit(
            intputs=project.kerasData[json_data["args"]["inputs"]],
            epochs=json_data["args"]["epochs"],
            batch_size=json_data["args"]["batch_size"],
        )
