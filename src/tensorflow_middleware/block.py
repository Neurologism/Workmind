# merging all functions into a single dict


type_to_function = {}

from .layer import layer_to_function
import inspect

for key in layer_to_function:
    type_to_function[key] = layer_to_function[key]

from .model import model_to_function

for key in model_to_function:
    type_to_function[key] = model_to_function[key]

from .dataset import dataset_to_function

for key in dataset_to_function:
    type_to_function[key] = dataset_to_function[key]

from .visualizer import visualizer_to_function

for key in visualizer_to_function:
    type_to_function[key] = visualizer_to_function[key]


# Block class
class Block:
    def __init__(self, params: dict, type: str) -> None:
        self.params = params
        if type not in type_to_function:
            raise ValueError(f"Type {type} not a known node type")
        self.function = type_to_function[type]
        self.object = None
        self.type = type

    def add_connection(
        self, connection_name: str, connection, connection_type: str
    ) -> None:
        if connection_name not in self.params:
            self.params[connection_name] = []
        self.params[connection_name].append([connection, connection_type])

    def __call__(self, **kwargs):
        if self.object is not None:
            return

        for key, value in kwargs.items():
            self.params[key] = value

        function_params = inspect.signature(self.function).parameters
        if len(function_params) == 1 and "params" in function_params:
            self.object = self.function(self.params)

        else:
            filtered_params = {
                k: v for k, v in self.params.items() if k in function_params
            }
            self.object = self.function(filtered_params)
