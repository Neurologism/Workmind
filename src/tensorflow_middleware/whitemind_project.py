from .dependencies import *
from .database_logger import DatabaseLogger
from .block import Block


class WhitemindProject:
    def __init__(
        self, json_data: dict | None = None, log_function=None, task_id="test"
    ) -> None:
        self.database_logger = DatabaseLogger(log_function)
        self.task_id = task_id

        self.blocks = {}

        # build the project objects out of blocks

        for node in json_data["nodes"]:
            additional_data = {
                "logger": self.database_logger,
                "task_id": task_id,
                "block_id": node["id"],
            }
            self.blocks[node["id"]] = Block(
                node["data"].update(additional_data) or additional_data,
                node["identifier"],
            )

        self.execution_head = self.blocks[json_data["start_node"]]

        for edge in json_data["edges"]:
            source_handle = edge["sourceHandle"].split("-")
            target_handle = edge["targetHandle"].split("-")

            if source_handle[0] == "val":
                source_handle = source_handle[1:]

            if target_handle[0] == "val":
                target_handle = target_handle[1:]

            source_block = self.blocks[source_handle[1]]
            target_block = self.blocks[target_handle[1]]

            source_block.add_connection(
                source_handle[0], target_block, target_handle[0]
            )
            target_block.add_connection(
                target_handle[0], source_block, source_handle[0]
            )

    def __call__(self):
        self.execution_head()

        # execute the project here with call functions of the helper classes
