from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.RandomCrop(
        height=operation["args"]["height"],
        width=operation["args"]["width"],
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
