from .dependencies import *

# customize layer functionality


def input_layer(params: dict):
    shape = params["dataset"].element_spec[0].shape[1:]

    params["shape"] = shape

    input_params = inspect.signature(keras.layers.Input).parameters
    input_params = {k: v for k, v in params.items() if k in input_params}

    return keras.layers.Input(**input_params)


def normalization_layer(params: dict):
    normalization_params = inspect.signature(keras.layers.Normalization).parameters
    normalization_params = {
        k: v for k, v in params.items() if k in normalization_params
    }

    normalization = keras.layers.Normalization(**normalization_params)
    normalization.adapt(params["dataset"].map(lambda x, y: x))

    return normalization


def dense_layer(params: dict):
    dense_params = inspect.signature(keras.layers.Dense).parameters
    dense_params = {k: v for k, v in params.items() if k in dense_params}

    previous_layer = params["in"][0][0].object
    input_shape = previous_layer.shape

    if len(input_shape) > 2:
        # Add a Flatten layer before the Dense layer
        flatten_layer = keras.layers.Flatten()(previous_layer)
        dense_layer = keras.layers.Dense(**dense_params)(flatten_layer)
    else:
        dense_layer = keras.layers.Dense(**dense_params)(previous_layer)

    return dense_layer


# map layer types to functions

layer_to_function = {
    "input": input_layer,
    "dense": dense_layer,
    "conv1d": keras.layers.Conv1D,
    "conv2d": keras.layers.Conv2D,
    "conv3d": keras.layers.Conv3D,
    "normalization": normalization_layer,
    "batch_normalization": keras.layers.BatchNormalization,
    "dropout": keras.layers.Dropout,
    "average_pooling1d": keras.layers.AveragePooling1D,
    "average_pooling2d": keras.layers.AveragePooling2D,
    "average_pooling3d": keras.layers.AveragePooling3D,
    "embedding": keras.layers.Embedding,
    "average": keras.layers.Average,
    "activation": keras.layers.Activation,
    "relu": keras.layers.ReLU,
}
