import json

from c_whitemind_model import WhitemindProject
import f_keras_factory


def read_json(file_path) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


a = WhitemindProject()
f_keras_factory.call(read_json("../../task.json"), a)
