import keras

from c_whitemind_model import WhitemindProject


def call(project: WhitemindProject) -> None:
    if project.json_data["method"] == "new":
        project.keras_data[project.json_data["uid"]] = keras.Model(
            project.keras_data[project.json_data["args"]["inputs"]],
            project.keras_data[project.json_data["args"]["outputs"]],
        )

    elif project.json_data["method"] == "compile":
        project.keras_data[project.json_data["uid"]].compile(
            optimizer=project.json_data["args"]["optimizer"]["method"],
            loss=project.json_data["args"]["loss"][0]["method"],
            metrics=project.json_data["args"]["metrics"][0]["method"],
        )

    elif project.json_data["method"] == "fit":
        project.keras_data[project.json_data["uid"]].fit(
            intputs=project.keras_data[project.json_data["args"]["inputs"]],
            epochs=project.json_data["args"]["epochs"],
            batch_size=project.json_data["args"]["batch_size"],
        )
