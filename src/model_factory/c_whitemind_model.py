import json

class WhitemindProject:
    def __init__(self, json_data: dict) -> None:
        self.json_data = json_data
        self.keras_data = {}

    def read_json(self, file_path) -> None:
        with open(file_path, "r") as file:
            self.json_data = json.load(file)

    from f_layer_factory import call as layer_factory_call
    from f_model_factory import call as f_model_factory_call

    def execute(self) -> None:
        for i in self.json_data["operations"]:
            datatype = i["type"]
            if datatype == "layer":
                self.layer_factory_call(self)

            if datatype == "model":
                self.f_model_factory_call(self)
