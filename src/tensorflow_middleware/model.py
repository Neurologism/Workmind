import keras
import inspect
import tensorflow as tf
import tf2onnx
import onnx


def topo_sort(self, nodes: dict) -> list:
    visited = set()
    stack = []

    def dfs(node: dict) -> None:
        if node["id"] in visited:
            return
        visited.add(node["id"])
        if "out" in node["data"]:
            for child_id in node["data"]["out"]:
                if child_id[0] in nodes:
                    dfs(nodes[child_id[0]])

        stack.append(node)

    for node in nodes.values():
        dfs(node)

    return stack[::-1]


def register_model(params: dict) -> keras.Model:
    # build layers
    sorted_layers = []

    def dfs(layer):
        if layer in sorted_layers:
            return

        if "out" in layer.params:
            for out in layer.params["out"]:
                dfs(out[0])

        sorted_layers.append(layer)

    for layer, connection in params["input"]:
        dfs(layer)

    sorted_layers = sorted_layers[::-1]

    for layer in sorted_layers:
        layer()

    model = keras.Model(params["input"][0][0].object, params["output"][0][0].object)

    compile_params = inspect.signature(model.compile).parameters
    compile_params = {k: v for k, v in params.items() if k in compile_params}

    model.compile(**compile_params)

    return model


def fit_model(params: dict) -> None:
    model = params["model"]

    fit_params = inspect.signature(model.fit).parameters
    fit_params = {k: v for k, v in params.items() if k in fit_params}

    model.fit(**fit_params)


def predict_model(params: dict) -> None:
    model = params["model"]

    predict_params = inspect.signature(model.predict).parameters
    predict_params = {k: v for k, v in params.items() if k in predict_params}

    model.predict(**predict_params)


def evaluate_model(params: dict) -> None:
    model = params["model"]

    evaluate_params = inspect.signature(model.evaluate).parameters
    evaluate_params = {k: v for k, v in params.items() if k in evaluate_params}

    model.evaluate(**evaluate_params)


def export_model(params: dict) -> None:
    model = params["model"]

    input_signature = [
        tf.TensorSpec(tensor.shape, tensor.dtype) for tensor in model.inputs
    ]

    onnx_model = tf2onnx.convert.from_keras(model, input_signature=input_signature)

    if isinstance(onnx_model, tuple):
        onnx_model = onnx_model[0]

    onnx.save_model(onnx_model, f"{params['output']}.onnx")


model_to_function = {
    "Model": register_model,
    "fit": fit_model,
    "predict": predict_model,
    "evaluate": evaluate_model,
    "export": export_model,
}
