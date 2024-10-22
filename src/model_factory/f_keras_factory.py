from c_whitemind_model import WhitemindProject
import f_layer_factory
import f_model_factory


def call(project: WhitemindProject) -> None:
    for i in project.json_data["operations"]:
        datatype = i["type"]
        if datatype == "layer":
            f_layer_factory.call(i, project)

        if datatype == "model":
            f_model_factory.call(i, project)
