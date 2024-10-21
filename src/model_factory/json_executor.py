import json
import f_keras_factory
from src.model_factory.c_whitemind_model import WhitemindProject


def read_json(file_path) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


a = WhitemindProject()
f_keras_factory.call(read_json("../../task.json"), a)
