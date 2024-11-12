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
        # case "Activation":
        #     f_activation_factory.call(self, operation)
        case "dense":
            f_dense_factory.call(self, layer)
        # case "EinsumDense":
        #     f_einsumdense_factory.call(self, operation)
        # case "Embedding":
        #     f_embedding_factory.call(self, operation)
        # case "Identity":
        #     f_identity_factory.call(self, operation)
        case "input":
            f_input_factory.call(self, layer)
        # case "InputSpec":
        #     f_inputspec_factory.call(self, operation)
        # case "Lambda":
        #     f_lambda_factory.call(self, operation)
        # case "Masking":
        #     f_masking_factory.call(self, operation)
        #
        # case "Conv1D":
        #     f_conv1d_factory.call(self, operation)
        # case "Conv1DTranspose":
        #     f_conv1dtranspose_factory.call(self, operation)
        # case "Conv2D":
        #     f_conv2d_factory.call(self, operation)
        # case "Conv2DTranspose":
        #     f_conv2dtranspose_factory.call(self, operation)
        # case "Conv3D":
        #     f_conv3d_factory.call(self, operation)
        # case "Conv3DTranspose":
        #     f_conv3dtranspose_factory.call(self, operation)
        # case "DepthwiseConv1D":
        #     f_depthwiseconv1d_factory.call(self, operation)
        # case "DepthwiseConv2D":
        #     f_depthwiseconv2d_factory.call(self, operation)
        # case "SeparableConv1D":
        #     f_separableconv1d_factory.call(self, operation)
        # case "SeparableConv2D":
        #     f_separableconv2d_factory.call(self, operation)
        #
        # case "AveragePooling1D":
        #     f_averagepooling1d_factory.call(self, operation)
        # case "AveragePooling2D":
        #     f_averagepooling2d_factory.call(self, operation)
        # case "AveragePooling3D":
        #     f_averagepooling3d_factory.call(self, operation)
        # case "GlobalAveragePooling1D":
        #     f_globalaveragepooling1d_factory.call(self, operation)
        # case "GlobalAveragePooling2D":
        #     f_globalaveragepooling2d_factory.call(self, operation)
        # case "GlobalAveragePooling3D":
        #     f_globalaveragepooling3d_factory.call(self, operation)
        # case "GlobalMaxPooling1D":
        #     f_globalmaxpooling1d_factory.call(self, operation)
        # case "GlobalMaxPooling2D":
        #     f_globalmaxpooling2d_factory.call(self, operation)
        # case "GlobalMaxPooling3D":
        #     f_globalmaxpooling3d_factory.call(self, operation)
        # case "MaxPooling1D":
        #     f_maxpooling1d_factory.call(self, operation)
        # case "MaxPooling2D":
        #     f_maxpooling2d_factory.call(self, operation)
        # case "MaxPooling3D":
        #     f_maxpooling3d_factory.call(self, operation)
        #
        # case "BaseRNN":
        #     f_basernn_factory.call(self, operation)
        # case "Bidirectional":
        #     f_bidirectional_factory.call(self, operation)
        # case "ConvLSTM1D":
        #     f_convlstm1d_factory.call(self, operation)
        # case "ConvLSTM2D":
        #     f_convlstm2d_factory.call(self, operation)
        # case "ConvLSTM3D":
        #     f_convlstm3d_factory.call(self, operation)
        # case "GRU":
        #     f_gru_factory.call(self, operation)
        # case "GRUCell":
        #     f_grucell_factory.call(self, operation)
        # case "LSTM":
        #     f_lstm_factory.call(self, operation)
        # case "LSTMCell":
        #     f_lstmcell_factory.call(self, operation)
        # case "SimpleRNN":
        #     f_simplernn_factory.call(self, operation)
        # case "SimpleRNNCell":
        #     f_simplernncell_factory.call(self, operation)
        # case "StackedRNNCell":
        #     f_stackedrnncell_factory.call(self, operation)
        # case "TimeDistributed":
        #     f_timedistributed_factory.call(self, operation)
        #
        # case "AutoContrast":
        #     f_autocontrast_factory.call(self, operation)
        # case "CategoryEncoding":
        #     f_categoryencoding_factory.call(self, operation)
        # case "CenterCrop":
        #     f_centercrop_factory.call(self, operation)
        # case "Discretization":
        #     f_discretization_factory.call(self, operation)
        # case "HashedCrossing":
        #     f_hashedcrossing_factory.call(self, operation)
        # case "Hashing":
        #     f_hasing_factory.call(self, operation)
        # case "IntegerLookup":
        #     f_integerlookup_factory.call(self, operation)
        # case "MelSpectrogram":
        #     f_melspectrogram_factory.call(self, operation)
        case "normalization":
            f_normalization_factory.call(self, layer)
        # case "Pipeline":
        #     f_pipeline_factory.call(self, operation)
        # case "RandomBrightness":
        #     f_randombrightness_factory.call(self, operation)
        # case "RandomContrast":
        #     f_randomcontrast_factory.call(self, operation)
        # case "RandomCrop":
        #     f_randomcrop_factory.call(self, operation)
        # case "RandomFlip":
        #     f_randomflip_factory.call(self, operation)
        # case "RandomRotation":
        #     f_randomrotation_factory.call(self, operation)
        # case "RandomTranslation":
        #     f_randomtranslation_factory.call(self, operation)
        # case "RandomZoom":
        #     f_randomzoom_factory.call(self, operation)
        # case "Rescaling":
        #     f_rescaling_factory.call(self, operation)
        # case "Resizing":
        #     f_resizing_factory.call(self, operation)
        # case "Solarization":
        #     f_solarization_factory.call(self, operation)
        # case "SpectralNormalization":
        #     f_spectralnormalization_factory.call(self, operation)
        # case "StringLookup":
        #     f_stringlookup_factory.call(self, operation)
        # case "TextVectorization":
        #     f_textvectorization_factory.call(self, operation)
        #
        # case "BatchNormalization":
        #     f_batchnormalization_factory.call(self, operation)
        # case "GroupNormalization":
        #     f_groupnormalization_factory.call(self, operation)
        # case "LayerNormalization":
        #     f_layernormalization_factory.call(self, operation)
        # case "UnitNormalization":
        #     f_unitnormalization_factory.call(self, operation)
        #
        # case "ActivityRegularization":
        #     f_activityregularization_factory.call(self, operation)
        # case "AlphaDropout":
        #     f_alphadropout_factory.call(self, operation)
        # case "Dropout":
        #     f_dropout_factory.call(self, operation)
        # case "GaussianDropout":
        #     f_gaussiandropout_factory.call(self, operation)
        # case "GaussianNoise":
        #     f_gaussiannoise_factory.call(self, operation)
        # case "SpatialDropout1D":
        #     f_spatialdropout1d_factory.call(self, operation)
        # case "SpatialDropout2D":
        #     f_spatialdropout2d_factory.call(self, operation)
        # case "SpatialDropout3D":
        #     f_spatialdropout3d_factory.call(self, operation)
        #
        # case "AdditiveAttention":
        #     f_additiveattention_factory.call(self, operation)
        # case "Attention":
        #     f_attention_factory.call(self, operation)
        # case "GroupQueryAttention":
        #     f_groupqueryattention_factory.call(self, operation)
        # case "MultiHeadAttention":
        #     f_multiheadattention_factory.call(self, operation)
        #
        # case "Cropping1D":
        #     f_cropping1d_factory.call(self, operation)
        # case "Cropping2D":
        #     f_cropping2d_factory.call(self, operation)
        # case "Cropping3D":
        #     f_cropping3d_factory.call(self, operation)
        case "flatten":
            f_flatten_factory.call(self, layer)
        # case "Permute":
        #     f_permute_factory.call(self, operation)
        # case "RepeatVector":
        #     f_repeatvector_factory.call(self, operation)
        # case "Reshape":
        #     f_reshape_factory.call(self, operation)
        # case "Upsampling1D":
        #     f_upsampling1d_factory.call(self, operation)
        # case "Upsampling2D":
        #     f_upsampling2d_factory.call(self, operation)
        # case "Upsampling3D":
        #     f_upsampling3d_factory.call(self, operation)
        # case "ZeroPadding1D":
        #     f_zeropadding1d_factory.call(self, operation)
        # case "ZeroPadding2D":
        #     f_zeropadding2d_factory.call(self, operation)
        # case "ZeroPadding3D":
        #     f_zeropadding3d_factory.call(self, operation)
        #
        # case "Add":
        #     f_add_factory.call(self, operation)
        # case "Average":
        #     f_average_factory.call(self, operation)
        # case "Concatenate":
        #     f_concatenate_factory.call(self, operation)
        # case "Dot":
        #     f_dot_factory.call(self, operation)
        # case "Maximum":
        #     f_maximum_factory.call(self, operation)
        # case "Minimum":
        #     f_minimum_factory.call(self, operation)
        # case "Multiply":
        #     f_multiply_factory.call(self, operation)
        # case "Subtract":
        #     f_subtract_factory.call(self, operation)
        #
        # case "ELU":
        #     f_elu_factory.call(self, operation)
        # case "LeakyReLU":
        #     f_leakyrelu_factory.call(self, operation)
        # case "PReLU":
        #     f_prelu_factory.call(self, operation)
        # case "ReLU":
        #     f_relu_factory.call(self, operation)
        # case "Softmax":
        #     f_softmax_factory.call(self, operation)

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
                if child_id in layers:
                    dfs(layers[child_id])

        stack.append(layer)

    for layer in layers.values():
        dfs(layer)

    return stack[::-1]


def call(self, layers: dict) -> None:
    sorted_layers = topo_sort(self, layers)
    for layer in sorted_layers:
        create(self, layer)
        inputs = layer["data"]["in"]
        layer_inputs = []
        for input_id in inputs:
            if input_id in layers:
                layer_inputs.append(self.project_data[input_id])

        if len(layer_inputs) == 1:
            self.project_data[layer["id"]] = self.project_data[layer["id"]](layer_inputs[0])

        elif len(layer_inputs) > 1:
            self.project_data[layer["id"]] = self.project_data[layer["id"]](layer_inputs)