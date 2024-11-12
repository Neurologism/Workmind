import json
import tensorflow as tf
from numpy.f2py.auxfuncs import throw_error


class WhitemindProject:
    def __init__(self, json_data: dict | None = None) -> None:
        if json_data is None:
            json_data = {}
        self.json_data = json_data
        self.project_data = {}

    def read_json(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            self.json_data = json.load(file)

    from .m_layer_factory import call as layer_factory_call
    from .m_model_factory import call as model_factory_call
    from .m_dataset_factory import call as dataset_factory_call
    from .m_initializer_factory import call as initializer_factory_call
    from .m_regularizer_factory import call as regularizer_factory_call
    from .m_constraint_factory import call as constraint_factory_call

    def execute(self) -> None:
        for operation in self.json_data["operations"]:
            match operation["type"]:
                case "layer":
                    self.layer_factory_call(operation)
                    inputs = self.search_input(operation["uid"])
                    if inputs:
                        self.project_data[operation["uid"]] = self.project_data[
                            operation["uid"]
                        ](inputs)

                case "model":
                    self.model_factory_call(operation)

                case "dataset":
                    self.dataset_factory_call(operation)

                case "initializer":
                    self.initializer_factory_call(operation)

                case "regularizer":
                    self.regularizer_factory_call(operation)

                case "constraint":
                    self.constraint_factory_call(operation)

                case _:
                    throw_error("Invalid class specified in operation")

    def search_input(self, name: str) -> list:
        inputs = []
        for operation in self.json_data["links"]:
            if operation["target"] == name:
                inputs.append(operation["source"])
        return inputs
