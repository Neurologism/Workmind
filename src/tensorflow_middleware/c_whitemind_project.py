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
        class_nodes = {"layer": [], "model": [], "dataset": []}

        for node in self.json_data["nodes"]:
            class_nodes[node["group_identifier"]].append(node)

        for node in class_nodes["dataset"]:
            self.dataset_factory_call(node)

        for node in class_nodes["layer"]:
            self.layer_factory_call(node)

        for node in class_nodes["model"]:
            self.model_factory_call(node)

    def search_input(self, name: str) -> list:
        inputs = []
        for operation in self.json_data["links"]:
            if operation["target"] == name:
                inputs.append(operation["source"])
        return inputs
