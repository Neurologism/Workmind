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
    # core layers
    "input": input_layer,
    "dense": dense_layer,
    "activation": keras.layers.Activation,
    "embedding": keras.layers.Embedding,
    # convolution layers
    "conv1d": keras.layers.Conv1D,
    "conv2d": keras.layers.Conv2D,
    "conv3d": keras.layers.Conv3D,
    # pooling layers
    "maxpooling1d": keras.layers.MaxPooling1D,
    "maxpooling2d": keras.layers.MaxPooling2D,
    "maxpooling3d": keras.layers.MaxPooling3D,
    "averagepooling1d": keras.layers.AveragePooling1D,
    "averagepooling2d": keras.layers.AveragePooling2D,
    "averagepooling3d": keras.layers.AveragePooling3D,
    "globalmaxpooling1d": keras.layers.GlobalMaxPooling1D,
    "globalmaxpooling2d": keras.layers.GlobalMaxPooling2D,
    "globalmaxpooling3d": keras.layers.GlobalMaxPooling3D,
    "globalaveragepooling1d": keras.layers.GlobalAveragePooling1D,
    "globalaveragepooling2d": keras.layers.GlobalAveragePooling2D,
    "globalaveragepooling3d": keras.layers.GlobalAveragePooling3D,
    # recurrent layers
    "lstm": keras.layers.LSTM,
    "gru": keras.layers.GRU,
    # preprocessing layers
    "normalization": normalization_layer,
    # normalization layers
    "batch_normalization": keras.layers.BatchNormalization,
    "layer_normalization": keras.layers.LayerNormalization,
    "unit_normalization": keras.layers.UnitNormalization,
    "group_normalization": keras.layers.GroupNormalization,
    # regularization layers
    "dropout": keras.layers.Dropout,
    "gaussian_dropout": keras.layers.GaussianDropout,
    "alpha_dropout": keras.layers.AlphaDropout,
    "gaussian_noise": keras.layers.GaussianNoise,
    "activity_regularization": keras.layers.ActivityRegularization,
    # attention layers
    "multi_head_attention": keras.layers.MultiHeadAttention,
    "attention": keras.layers.Attention,
    # reshaping layers
    "reshape": keras.layers.Reshape,
    "flatten": keras.layers.Flatten,
    "permute": keras.layers.Permute,
    "zero_padding1d": keras.layers.ZeroPadding1D,
    "zero_padding2d": keras.layers.ZeroPadding2D,
    "zero_padding3d": keras.layers.ZeroPadding3D,
    # merging layers
    "concatenate": keras.layers.Concatenate,
    "average": keras.layers.Average,
    "maximum": keras.layers.Maximum,
    "minimum": keras.layers.Minimum,
    "add": keras.layers.Add,
    "subtract": keras.layers.Subtract,
    "multiply": keras.layers.Multiply,
    "dot": keras.layers.Dot,
    # activation layers
    "relu": keras.layers.ReLU,
    "softmax": keras.layers.Softmax,
    "leaky_relu": keras.layers.LeakyReLU,
    "prelu": keras.layers.PReLU,
    "elu": keras.layers.ELU,
}
