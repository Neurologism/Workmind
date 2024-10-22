import json

class WhitemindProject:
    def __init__(self, json_data=None) -> None:
        if json_data is None:
            json_data = {}
        self.json_data = json_data
        self.keras_data = {}

    def read_json(self, file_path) -> None:
        with open(file_path, "r") as file:
            self.json_data = json.load(file)

    from m_layer_factory import call as layer_factory_call
    from m_model_factory import call as f_model_factory_call

    def execute(self) -> None:
        for operation in self.json_data["operations"]:
            if operation["type"] == "layer":
                self.layer_factory_call(operation)

            if operation["type"] == "model":
                self.f_model_factory_call(operation)


a = WhitemindProject()
a.read_json("../../task.json")
a.execute()
