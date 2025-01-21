from tensorflow_middleware import WhitemindProject
import json

if __name__ == "__main__":
    with open("task.json", "r") as file:
        json_data = json.load(file)
    WhitemindProject(json_data)
