import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        self.project_data[operation["uid"]] = keras.Model(
            self.project_data[operation["args"]["inputs"]],
            self.project_data[operation["args"]["outputs"]],
        )

    elif operation["method"] == "compile":
        self.project_data[operation["uid"]].compile(
            optimizer=operation["args"]["optimizer"]["method"],
            loss=operation["args"]["loss"][0]["method"],
            metrics=[operation["args"]["metrics"][0]["method"]],
        )

    elif operation["method"] == "fit":

        self.project_data[operation["uid"]].fit(
            x=self.project_data[operation["args"]["dataset"]],
            epochs=operation["args"]["epochs"],
        )

    elif operation["method"] == "evaluate":
        self.project_data[operation["uid"]].evaluate(
            x=self.project_data[operation["args"]["dataset"]],
        )
