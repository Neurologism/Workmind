from ...m_dependencies import *


def call(self, operation: dict) -> None:
    example = next(
        iter(
            self.project_data[operation["data"]["dataset"][0]][
                operation["data"]["dataset"][1]
            ]
        )
    )[0]
    self.project_data[operation["id"]] = keras.layers.Input(
        shape=example.shape[1:],
        batch_size=(
            operation["data"]["batch_size"] if operation["data"]["batch_size"] else None
        ),
        dtype=(operation["data"]["dtype"] if "dtype" in operation["data"] else None),
        sparse=(
            operation["data"]["sparse"] if "sparse" in operation["data"] else False
        ),
        batch_shape=(
            tuple(x for x in operation["data"]["batch_shape"])
            if "batch_shape" in operation["data"]
            else None
        ),
        name=(operation["data"]["name"] if "name" in operation["data"] else None),
        tensor=(
            self.project_data[operation["data"]["tensor"]]
            if "tensor" in operation["data"]
            else None
        ),
        optional=(
            operation["data"]["optional"] if "optional" in operation["data"] else False
        ),
    )
