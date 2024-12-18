from .m_dependencies import *
from .m_layer_factory import call as layer_factory_call
from .m_model_factory import call as model_factory_call
from .m_dataset_factory import call as dataset_factory_call
from .m_initializer_factory import call as initializer_factory_call
from .m_regularizer_factory import call as regularizer_factory_call
from .m_constraint_factory import call as constraint_factory_call
from .c_callbacks import DatabaseLogger
from .m_visualizer_factory import call as visualizer_factory_call


class WhitemindProject:
    def __init__(self, json_data: dict | None = None, log_function=None) -> None:
        if json_data is None:
            json_data = {}
        self.json_data = json_data
        self.project_data = {}
        self.callbacks = [DatabaseLogger(log_function, self)] if log_function else []

    def read_json(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            self.json_data = json.load(file)

    def execute(self) -> None:
        class_nodes = {"layer": {}, "model": {}, "dataset": {}, "visualizer": {}}
        group_map = {}

        # sort nodes by group
        for node in self.json_data["nodes"]:
            class_nodes[node["group_identifier"]][node["id"]] = node
            group_map[node["id"]] = node["group_identifier"]

        # store convert edge list to attributes of nodes
        for edge in self.json_data["edges"]:
            source_handle = edge["sourceHandle"].split("-")
            target_handle = edge["targetHandle"].split("-")

            if source_handle[0] == "val":
                source_handle = source_handle[1:]

            if target_handle[0] == "val":
                target_handle = target_handle[1:]

            # create empty data attribute if not present
            if "data" not in class_nodes[group_map[source_handle[1]]][source_handle[1]]:
                class_nodes[group_map[source_handle[1]]][source_handle[1]]["data"] = {}

            if "data" not in class_nodes[group_map[target_handle[1]]][target_handle[1]]:
                class_nodes[group_map[target_handle[1]]][target_handle[1]]["data"] = {}

            # create empty list if attribute is not present
            if (
                source_handle[0]
                not in class_nodes[group_map[source_handle[1]]][source_handle[1]][
                    "data"
                ]
            ):
                class_nodes[group_map[source_handle[1]]][source_handle[1]]["data"][
                    source_handle[0]
                ] = []

            if (
                target_handle[0]
                not in class_nodes[group_map[target_handle[1]]][target_handle[1]][
                    "data"
                ]
            ):
                class_nodes[group_map[target_handle[1]]][target_handle[1]]["data"][
                    target_handle[0]
                ] = []

            # create a list of edges for each node containing the other node and the attribute
            class_nodes[group_map[source_handle[1]]][source_handle[1]]["data"][
                source_handle[0]
            ].append([target_handle[1], target_handle[0]])
            class_nodes[group_map[target_handle[1]]][target_handle[1]]["data"][
                target_handle[0]
            ].append([source_handle[1], source_handle[0]])

        # ATTENTION order of execution is important
        dataset_factory_call(self, class_nodes["dataset"])

        layer_factory_call(self, class_nodes["layer"])

        visualizer_factory_call(self, class_nodes["visualizer"])

        model_factory_call(self, class_nodes["model"])

    def search_input(self, name: str) -> list:
        inputs = []
        for operation in self.json_data["links"]:
            if operation["target"] == name:
                inputs.append(operation["source"])
        return inputs
