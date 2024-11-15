from tensorflow_middleware import WhitemindProject

if __name__ == "__main__":
    a = WhitemindProject()
    a.read_json("task.json")
    a.execute()
