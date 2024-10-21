import f_dense_factory
import f_input_factory
from src.model_factory.c_whitemind_model import WhitemindProject


def new(json_data: dict, project: WhitemindProject) -> None:
    if json_data["args"]["class"] == "Dense":
        f_dense_factory.call(json_data, project)

    if json_data["args"]["class"] == "Input":
        f_input_factory.call(json_data, project)


def call(json_data: dict, project: WhitemindProject) -> None:
    if json_data["method"] == "new":
        new(json_data, project)
