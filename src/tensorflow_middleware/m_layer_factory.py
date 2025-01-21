from .layer_factories.core_factories import f_activation_factory
from .layer_factories.core_factories import f_activation_factory
from .layer_factories.core_factories import f_dense_factory
from .layer_factories.core_factories import f_einsumdense_factory
from .layer_factories.core_factories import f_embedding_factory
from .layer_factories.core_factories import f_identity_factory
from .layer_factories.core_factories import f_input_factory
from .layer_factories.core_factories import f_inputspec_factory
from .layer_factories.core_factories import f_lambda_factory
from .layer_factories.core_factories import f_masking_factory

from .layer_factories.convolution_factories import f_conv1d_factory
from .layer_factories.convolution_factories import f_conv1dtranspose_factory
from .layer_factories.convolution_factories import f_conv2d_factory
from .layer_factories.convolution_factories import f_conv2dtranspose_factory
from .layer_factories.convolution_factories import f_conv3d_factory
from .layer_factories.convolution_factories import f_conv3dtranspose_factory
from .layer_factories.convolution_factories import f_depthwiseconv1d_factory
from .layer_factories.convolution_factories import f_depthwiseconv2d_factory
from .layer_factories.convolution_factories import f_separableconv1d_factory
from .layer_factories.convolution_factories import f_separableconv2d_factory

from .layer_factories.pooling_factories import f_averagepooling1d_factory
from .layer_factories.pooling_factories import f_averagepooling2d_factory
from .layer_factories.pooling_factories import f_averagepooling3d_factory
from .layer_factories.pooling_factories import f_globalaveragepooling1d_factory
from .layer_factories.pooling_factories import f_globalaveragepooling2d_factory
from .layer_factories.pooling_factories import f_globalaveragepooling3d_factory
from .layer_factories.pooling_factories import f_globalmaxpooling1d_factory
from .layer_factories.pooling_factories import f_globalmaxpooling2d_factory
from .layer_factories.pooling_factories import f_globalmaxpooling3d_factory
from .layer_factories.pooling_factories import f_maxpooling1d_factory
from .layer_factories.pooling_factories import f_maxpooling2d_factory
from .layer_factories.pooling_factories import f_maxpooling3d_factory

from .layer_factories.recurrent_factories import f_basernn_factory
from .layer_factories.recurrent_factories import f_bidirectional_factory
from .layer_factories.recurrent_factories import f_convlstm1d_factory
from .layer_factories.recurrent_factories import f_convlstm2d_factory
from .layer_factories.recurrent_factories import f_convlstm3d_factory
from .layer_factories.recurrent_factories import f_gru_factory
from .layer_factories.recurrent_factories import f_grucell_factory
from .layer_factories.recurrent_factories import f_lstm_factory
from .layer_factories.recurrent_factories import f_lstmcell_factory
from .layer_factories.recurrent_factories import f_simplernn_factory
from .layer_factories.recurrent_factories import f_simplernncell_factory
from .layer_factories.recurrent_factories import f_stackedrnncell_factory
from .layer_factories.recurrent_factories import f_timedistributed_factory

from .layer_factories.preprocessing_factories import f_autocontrast_factory
from .layer_factories.preprocessing_factories import f_categoryencoding_factory
from .layer_factories.preprocessing_factories import f_centercrop_factory
from .layer_factories.preprocessing_factories import f_discretization_factory
from .layer_factories.preprocessing_factories import f_hashedcrossing_factory
from .layer_factories.preprocessing_factories import f_hasing_factory
from .layer_factories.preprocessing_factories import f_integerlookup_factory
from .layer_factories.preprocessing_factories import f_melspectrogram_factory
from .layer_factories.preprocessing_factories import f_normalization_factory
from .layer_factories.preprocessing_factories import f_pipeline_factory
from .layer_factories.preprocessing_factories import f_randombrightness_factory
from .layer_factories.preprocessing_factories import f_randomcontrast_factory
from .layer_factories.preprocessing_factories import f_randomcrop_factory
from .layer_factories.preprocessing_factories import f_randomflip_factory
from .layer_factories.preprocessing_factories import f_randomrotation_factory
from .layer_factories.preprocessing_factories import f_randomtranslation_factory
from .layer_factories.preprocessing_factories import f_randomzoom_factory
from .layer_factories.preprocessing_factories import f_rescaling_factory
from .layer_factories.preprocessing_factories import f_resizing_factory
from .layer_factories.preprocessing_factories import f_solarization_factory
from .layer_factories.preprocessing_factories import f_spectralnormalization_factory
from .layer_factories.preprocessing_factories import f_stringlookup_factory
from .layer_factories.preprocessing_factories import f_textvectorization_factory

from .layer_factories.normalization_factories import f_batchnormalization_factory
from .layer_factories.normalization_factories import f_groupnormalization_factory
from .layer_factories.normalization_factories import f_layernormalization_factory
from .layer_factories.normalization_factories import f_unitnormalization_factory

from .layer_factories.regularization_factories import f_activityregularization_factory
from .layer_factories.regularization_factories import f_alphadropout_factory
from .layer_factories.regularization_factories import f_dropout_factory
from .layer_factories.regularization_factories import f_gaussiandropout_factory
from .layer_factories.regularization_factories import f_gaussiannoise_factory
from .layer_factories.regularization_factories import f_spatialdropout1d_factory
from .layer_factories.regularization_factories import f_spatialdropout2d_factory
from .layer_factories.regularization_factories import f_spatialdropout3d_factory

from .layer_factories.attention_factories import f_additiveattention_factory
from .layer_factories.attention_factories import f_attention_factory
from .layer_factories.attention_factories import f_groupqueryattention_factory
from .layer_factories.attention_factories import f_multiheadattention_factory

from .layer_factories.reshaping_factories import f_cropping1d_factory
from .layer_factories.reshaping_factories import f_cropping2d_factory
from .layer_factories.reshaping_factories import f_cropping3d_factory
from .layer_factories.reshaping_factories import f_flatten_factory
from .layer_factories.reshaping_factories import f_permute_factory
from .layer_factories.reshaping_factories import f_repeatvector_factory
from .layer_factories.reshaping_factories import f_reshape_factory
from .layer_factories.reshaping_factories import f_upsampling1d_factory
from .layer_factories.reshaping_factories import f_upsampling2d_factory
from .layer_factories.reshaping_factories import f_upsampling3d_factory
from .layer_factories.reshaping_factories import f_zeropadding1d_factory
from .layer_factories.reshaping_factories import f_zeropadding2d_factory
from .layer_factories.reshaping_factories import f_zeropadding3d_factory

from .layer_factories.merging_factories import f_add_factory
from .layer_factories.merging_factories import f_average_factory
from .layer_factories.merging_factories import f_concatenate_factory
from .layer_factories.merging_factories import f_dot_factory
from .layer_factories.merging_factories import f_maximum_factory
from .layer_factories.merging_factories import f_minimum_factory
from .layer_factories.merging_factories import f_multiply_factory
from .layer_factories.merging_factories import f_subtract_factory

from .layer_factories.activation_factories import f_elu_factory
from .layer_factories.activation_factories import f_leakyrelu_factory
from .layer_factories.activation_factories import f_prelu_factory
from .layer_factories.activation_factories import f_relu_factory
from .layer_factories.activation_factories import f_softmax_factory


def create(self, layer: dict) -> None:
    match layer["identifier"]:
        case "relu":
            f_relu_factory.call(self, layer)
        # case "Softmax":
        #     f_softmax_factory.call(self, layer)

        case _:
            raise ValueError(f"Layer class {layer["identifier"]} not supported")


def topo_sort(self, layers: dict) -> list:
    visited = set()
    stack = []

    def dfs(layer: dict) -> None:
        if layer["id"] in visited:
            return
        visited.add(layer["id"])
        if "out" in layer["data"]:
            for child_id in layer["data"]["out"]:
                if child_id[0] in layers:
                    dfs(layers[child_id[0]])

        stack.append(layer)

    for layer in layers.values():
        dfs(layer)

    return stack[::-1]


def call(self, layers: dict, models: dict) -> None:
    sorted_layers = topo_sort(self, layers)
    i = 0
    for layer in sorted_layers:
        for input in layer["data"]["in"]:
            if input[0] in models:
                if "dataset" in models[input[0]]["data"]:
                    sorted_layers[i]["data"]["dataset"] = models[input[0]]["data"][
                        "dataset"
                    ]
            elif input[0] in layers:
                if "dataset" in layers[input[0]]["data"]:
                    sorted_layers[i]["data"]["dataset"] = layers[input[0]]["data"][
                        "dataset"
                    ]
        i += 1

    for layer in sorted_layers:
        if "dataset" not in layer["data"]:
            continue
        create(self, layer)
        inputs = layer["data"]["in"]
        layer_inputs = []
        for input_id in inputs:
            if input_id[0] in layers:
                layer_inputs.append(self.project_data[input_id[0]])

        if len(layer_inputs) == 1:
            self.project_data[layer["id"]] = self.project_data[layer["id"]](
                layer_inputs[0]
            )

        elif len(layer_inputs) > 1:
            self.project_data[layer["id"]] = self.project_data[layer["id"]](
                layer_inputs
            )
