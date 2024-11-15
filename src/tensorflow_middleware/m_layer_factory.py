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
        case "Activation":
            f_activation_factory.call(self, layer)
        case "dense":
            f_dense_factory.call(self, layer)
        # case "EinsumDense":
        #     f_einsumdense_factory.call(self, layer)
        case "Embedding":
            f_embedding_factory.call(self, layer)
        # case "Identity":
        #     f_identity_factory.call(self, layer)
        case "input":
            f_input_factory.call(self, layer)
        # case "InputSpec":
        #     f_inputspec_factory.call(self, layer)
        # case "Lambda":
        #     f_lambda_factory.call(self, layer)
        # case "Masking":
        #     f_masking_factory.call(self, layer)
        #
        case "Conv1D":
            f_conv1d_factory.call(self, layer)
        # case "Conv1DTranspose":
        #     f_conv1dtranspose_factory.call(self, layer)
        case "Conv2D":
            f_conv2d_factory.call(self, layer)
        # case "Conv2DTranspose":
        #     f_conv2dtranspose_factory.call(self, layer)
        case "Conv3D":
            f_conv3d_factory.call(self, layer)
        # case "Conv3DTranspose":
        #     f_conv3dtranspose_factory.call(self, layer)
        # case "DepthwiseConv1D":
        #     f_depthwiseconv1d_factory.call(self, layer)
        # case "DepthwiseConv2D":
        #     f_depthwiseconv2d_factory.call(self, layer)
        # case "SeparableConv1D":
        #     f_separableconv1d_factory.call(self, layer)
        # case "SeparableConv2D":
        #     f_separableconv2d_factory.call(self, layer)
        #
        # case "AveragePooling1D":
        #     f_averagepooling1d_factory.call(self, layer)
        # case "AveragePooling2D":
        #     f_averagepooling2d_factory.call(self, layer)
        # case "AveragePooling3D":
        #     f_averagepooling3d_factory.call(self, layer)
        # case "GlobalAveragePooling1D":
        #     f_globalaveragepooling1d_factory.call(self, layer)
        # case "GlobalAveragePooling2D":
        #     f_globalaveragepooling2d_factory.call(self, layer)
        # case "GlobalAveragePooling3D":
        #     f_globalaveragepooling3d_factory.call(self, layer)
        # case "GlobalMaxPooling1D":
        #     f_globalmaxpooling1d_factory.call(self, layer)
        # case "GlobalMaxPooling2D":
        #     f_globalmaxpooling2d_factory.call(self, layer)
        # case "GlobalMaxPooling3D":
        #     f_globalmaxpooling3d_factory.call(self, layer)
        # case "MaxPooling1D":
        #     f_maxpooling1d_factory.call(self, layer)
        # case "MaxPooling2D":
        #     f_maxpooling2d_factory.call(self, layer)
        # case "MaxPooling3D":
        #     f_maxpooling3d_factory.call(self, layer)
        #
        # case "BaseRNN":
        #     f_basernn_factory.call(self, layer)
        # case "Bidirectional":
        #     f_bidirectional_factory.call(self, layer)
        # case "ConvLSTM1D":
        #     f_convlstm1d_factory.call(self, layer)
        # case "ConvLSTM2D":
        #     f_convlstm2d_factory.call(self, layer)
        # case "ConvLSTM3D":
        #     f_convlstm3d_factory.call(self, layer)
        # case "GRU":
        #     f_gru_factory.call(self, layer)
        # case "GRUCell":
        #     f_grucell_factory.call(self, layer)
        # case "LSTM":
        #     f_lstm_factory.call(self, layer)
        # case "LSTMCell":
        #     f_lstmcell_factory.call(self, layer)
        # case "SimpleRNN":
        #     f_simplernn_factory.call(self, layer)
        # case "SimpleRNNCell":
        #     f_simplernncell_factory.call(self, layer)
        # case "StackedRNNCell":
        #     f_stackedrnncell_factory.call(self, layer)
        # case "TimeDistributed":
        #     f_timedistributed_factory.call(self, layer)
        #
        # case "AutoContrast":
        #     f_autocontrast_factory.call(self, layer)
        # case "CategoryEncoding":
        #     f_categoryencoding_factory.call(self, layer)
        # case "CenterCrop":
        #     f_centercrop_factory.call(self, layer)
        # case "Discretization":
        #     f_discretization_factory.call(self, layer)
        # case "HashedCrossing":
        #     f_hashedcrossing_factory.call(self, layer)
        # case "Hashing":
        #     f_hasing_factory.call(self, layer)
        # case "IntegerLookup":
        #     f_integerlookup_factory.call(self, layer)
        # case "MelSpectrogram":
        #     f_melspectrogram_factory.call(self, layer)
        case "normalization":
            f_normalization_factory.call(self, layer)
        # case "Pipeline":
        #     f_pipeline_factory.call(self, layer)
        # case "RandomBrightness":
        #     f_randombrightness_factory.call(self, layer)
        # case "RandomContrast":
        #     f_randomcontrast_factory.call(self, layer)
        # case "RandomCrop":
        #     f_randomcrop_factory.call(self, layer)
        # case "RandomFlip":
        #     f_randomflip_factory.call(self, layer)
        # case "RandomRotation":
        #     f_randomrotation_factory.call(self, layer)
        # case "RandomTranslation":
        #     f_randomtranslation_factory.call(self, layer)
        # case "RandomZoom":
        #     f_randomzoom_factory.call(self, layer)
        # case "Rescaling":
        #     f_rescaling_factory.call(self, layer)
        # case "Resizing":
        #     f_resizing_factory.call(self, layer)
        # case "Solarization":
        #     f_solarization_factory.call(self, layer)
        # case "SpectralNormalization":
        #     f_spectralnormalization_factory.call(self, layer)
        # case "StringLookup":
        #     f_stringlookup_factory.call(self, layer)
        # case "TextVectorization":
        #     f_textvectorization_factory.call(self, layer)
        #
        case "BatchNormalization":
            f_batchnormalization_factory.call(self, layer)
        # case "GroupNormalization":
        #     f_groupnormalization_factory.call(self, layer)
        # case "LayerNormalization":
        #     f_layernormalization_factory.call(self, layer)
        # case "UnitNormalization":
        #     f_unitnormalization_factory.call(self, layer)
        #
        # case "ActivityRegularization":
        #     f_activityregularization_factory.call(self, layer)
        # case "AlphaDropout":
        #     f_alphadropout_factory.call(self, layer)
        case "Dropout":
            f_dropout_factory.call(self, layer)
        # case "GaussianDropout":
        #     f_gaussiandropout_factory.call(self, layer)
        # case "GaussianNoise":
        #     f_gaussiannoise_factory.call(self, layer)
        # case "SpatialDropout1D":
        #     f_spatialdropout1d_factory.call(self, layer)
        # case "SpatialDropout2D":
        #     f_spatialdropout2d_factory.call(self, layer)
        # case "SpatialDropout3D":
        #     f_spatialdropout3d_factory.call(self, layer)
        #
        # case "AdditiveAttention":
        #     f_additiveattention_factory.call(self, layer)
        # case "Attention":
        #     f_attention_factory.call(self, layer)
        # case "GroupQueryAttention":
        #     f_groupqueryattention_factory.call(self, layer)
        # case "MultiHeadAttention":
        #     f_multiheadattention_factory.call(self, layer)
        #
        # case "Cropping1D":
        #     f_cropping1d_factory.call(self, layer)
        # case "Cropping2D":
        #     f_cropping2d_factory.call(self, layer)
        # case "Cropping3D":
        #     f_cropping3d_factory.call(self, layer)
        case "flatten":
            f_flatten_factory.call(self, layer)
        # case "Permute":
        #     f_permute_factory.call(self, layer)
        # case "RepeatVector":
        #     f_repeatvector_factory.call(self, layer)
        # case "Reshape":
        #     f_reshape_factory.call(self, layer)
        # case "Upsampling1D":
        #     f_upsampling1d_factory.call(self, layer)
        # case "Upsampling2D":
        #     f_upsampling2d_factory.call(self, layer)
        # case "Upsampling3D":
        #     f_upsampling3d_factory.call(self, layer)
        # case "ZeroPadding1D":
        #     f_zeropadding1d_factory.call(self, layer)
        # case "ZeroPadding2D":
        #     f_zeropadding2d_factory.call(self, layer)
        # case "ZeroPadding3D":
        #     f_zeropadding3d_factory.call(self, layer)
        #
        # case "Add":
        #     f_add_factory.call(self, layer)
        case "Average":
            f_average_factory.call(self, layer)
        # case "Concatenate":
        #     f_concatenate_factory.call(self, layer)
        # case "Dot":
        #     f_dot_factory.call(self, layer)
        # case "Maximum":
        #     f_maximum_factory.call(self, layer)
        # case "Minimum":
        #     f_minimum_factory.call(self, layer)
        # case "Multiply":
        #     f_multiply_factory.call(self, layer)
        # case "Subtract":
        #     f_subtract_factory.call(self, layer)
        #
        # case "ELU":
        #     f_elu_factory.call(self, layer)
        # case "LeakyReLU":
        #     f_leakyrelu_factory.call(self, layer)
        # case "PReLU":
        #     f_prelu_factory.call(self, layer)
        case "ReLU":
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
            self.project_data[layer["id"]] = self.project_data[layer["id"]](
                layer_inputs[0]
            )

        elif len(layer_inputs) > 1:
            self.project_data[layer["id"]] = self.project_data[layer["id"]](
                layer_inputs
            )
