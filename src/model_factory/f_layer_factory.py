from c_whitemind_model import WhitemindProject
import f_dense_factory
import f_input_factory


def new(project: WhitemindProject) -> None:
    if project.json_data["args"]["class"] == "Dense":
        f_dense_factory.call(project.json_data, project)

    if project.json_data["args"]["class"] == "Input":
        f_input_factory.call(project.json_data, project)


def call(project: WhitemindProject) -> None:
    if project.json_data["method"] == "new":
        new(project.json_data, project)
