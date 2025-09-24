from .dependencies import *


def register_model(params: dict) -> keras.Model:
    # build layers
    sorted_layers = []

    def dfs(layer):
        if layer in sorted_layers:
            return

        if layer.type == "Model":
            return

        if "out" in layer.params:
            for out in layer.params["out"]:
                dfs(out[0])

        sorted_layers.append(layer)

    for layer, connection in params["input"]:
        dfs(layer)

    sorted_layers = sorted_layers[::-1]

    def get_dataset(layer):
        if layer.type == "fit":
            return [layer.params["data"][0][0], layer.params["data"][0][1]]

        for out in layer.params["out"]:
            dataset = get_dataset(out[0])
            if dataset:
                return dataset

        return None

    dataset, shard = get_dataset(params["output"][0][0])

    if dataset is None:
        raise ValueError("No dataset found for this model")

    dataset()

    for layer in sorted_layers:
        layer(**{"dataset": dataset.object[shard]})

        if callable(layer.object):
            layer_inputs = []
            for input in layer.params["in"]:
                layer_inputs.append(input[0].object)

            if len(layer_inputs) == 1:
                layer.object = layer.object(layer_inputs[0])

            elif len(layer_inputs) > 1:
                layer.object = layer.object(layer_inputs)

    model = keras.Model(params["input"][0][0].object, params["output"][0][0].object)

    compile_params = inspect.signature(model.compile).parameters
    compile_params = {k: v for k, v in params.items() if k in compile_params}

    model.compile(**compile_params)

    if "out" in params:
        params["out"][0][0](**{"model": model})

    return model


def fit_model(params: dict):
    params["logger"].block_payloads[params["block_id"]] = [
        {
            "loss": "loss",
            "val_loss": "val_loss",
        }
    ]
    model = params["model"]

    callbacks = [params["logger"]]

    if "early_stopping" in params and params["early_stopping"]:
        callbacks.append(
            keras.callbacks.EarlyStopping(patience=params["early_stopping"])
        )

    params["callbacks"] = callbacks
    params["x"] = params["data"][0][0].object[params["data"][0][1]]
    if "validation_data" in params:
        params["validation_data"] = params["validation_data"][0][0].object[
            params["validation_data"][0][1]
        ]

    if "visualizers" in params:
        for visualizer, connection in params["visualizers"]:
            visualizer()

    fit_params = inspect.signature(model.fit).parameters
    fit_params = {k: v for k, v in params.items() if k in fit_params}

    model.fit(**fit_params)

    if "out" in params:
        params["out"][0][0](**{"model": model})

    return model


def predict_model(params: dict):
    model = params["model"]

    callbacks = [params["logger"]]

    params["callbacks"] = callbacks

    predict_params = inspect.signature(model.predict).parameters
    predict_params = {k: v for k, v in params.items() if k in predict_params}

    model.predict(**predict_params)

    if "out" in params:
        params["out"][0][0](**{"model": model})

    return model


def evaluate_model(params: dict):
    model = params["model"]

    callbacks = [params["logger"]]

    params["callbacks"] = callbacks

    evaluate_params = inspect.signature(model.evaluate).parameters
    evaluate_params = {k: v for k, v in params.items() if k in evaluate_params}

    model.evaluate(**evaluate_params)

    if "out" in params:
        params["out"][0][0](**{"model": model})

    return model


# def export_model(params: dict):
#     model = params["model"]

#     input_signature = [
#         tf.TensorSpec(tensor.shape, tensor.dtype) for tensor in model.inputs
#     ]

#     onnx_model = tf2onnx.convert.from_keras(model, input_signature=input_signature)

#     if isinstance(onnx_model, tuple):
#         onnx_model = onnx_model[0]

#     onnx.save_model(onnx_model, f"{params['task_id']}.onnx")

#     s3 = boto3.client(
#         "s3",
#         config=boto3.session.Config(signature_version="s3v4"),
#         region_name="eu-central-1",
#     )
#     with open(f"{params['task_id']}.onnx", "rb") as f:
#         s3.upload_fileobj(f, "whitemind-models", f"{params['task_id']}.onnx")

#     url = s3.generate_presigned_url(
#         "get_object",
#         Params={"Bucket": "whitemind-models", "Key": f"{params['task_id']}.onnx"},
#     )

#     params["logger"].on_export(params["block_id"],url)

#     if "out" in params:
#         params["out"][0][0](**{"model": model})

#     return model


model_to_function = {
    "Model": register_model,
    "fit": fit_model,
    "predict": predict_model,
    "evaluate": evaluate_model,
    "export": export_model,
}
