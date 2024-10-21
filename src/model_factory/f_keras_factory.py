import f_layer_factory
import f_model_factory
from c_whitemind_model import WhitemindProject


def call(json_data: dict, project: WhitemindProject) -> None:
    for i in json_data["operations"]:
        datatype = i["type"]
        if datatype == "layer":
            f_layer_factory.call(i, project)

        if datatype == "model":
            f_model_factory.call(i, project)
