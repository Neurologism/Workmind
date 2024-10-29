import layer_factories.core_factories.f_activation_factory as f_activation_factory
import layer_factories.core_factories.f_dense_factory as f_dense_factory
import layer_factories.core_factories.f_einsumdense_factory as f_einsumdense_factory
import layer_factories.core_factories.f_embedding_factory as f_embedding_factory
import layer_factories.core_factories.f_identity_factory as f_identity_factory
import layer_factories.core_factories.f_input_factory as f_input_factory
import layer_factories.core_factories.f_inputspec_factory as f_inputspec_factory
import layer_factories.core_factories.f_lambda_factory as f_lambda_factory
import layer_factories.core_factories.f_masking_factory as f_masking_factory

import layer_factories.convolution_factories.f_conv1d_factory as f_conv1d_factory
import layer_factories.convolution_factories.f_conv1dtranspose_factory as f_conv1dtranspose_factory
import layer_factories.convolution_factories.f_conv2d_factory as f_conv2d_factory
import layer_factories.convolution_factories.f_conv2dtranspose_factory as f_conv2dtranspose_factory
import layer_factories.convolution_factories.f_conv3d_factory as f_conv3d_factory
import layer_factories.convolution_factories.f_conv3dtranspose_factory as f_conv3dtranspose_factory
import layer_factories.convolution_factories.f_depthwiseconv1d_factory as f_depthwiseconv1d_factory
import layer_factories.convolution_factories.f_depthwiseconv2d_factory as f_depthwiseconv2d_factory
import layer_factories.convolution_factories.f_separableconv1d_factory as f_separableconv1d_factory
import layer_factories.convolution_factories.f_separableconv2d_factory as f_separableconv2d_factory

import layer_factories.pooling_factories.f_averagepooling1d_factory as f_averagepooling1d_factory
import layer_factories.pooling_factories.f_averagepooling2d_factory as f_averagepooling2d_factory
import layer_factories.pooling_factories.f_averagepooling3d_factory as f_averagepooling3d_factory
import layer_factories.pooling_factories.f_globalaveragepooling1d_factory as f_globalaveragepooling1d_factory
import layer_factories.pooling_factories.f_globalaveragepooling2d_factory as f_globalaveragepooling2d_factory
import layer_factories.pooling_factories.f_globalaveragepooling3d_factory as f_globalaveragepooling3d_factory
import layer_factories.pooling_factories.f_globalmaxpooling1d_factory as f_globamaxpooling1d_factory
import layer_factories.pooling_factories.f_globalmaxpooling2d_factory as f_globamaxpooling2d_factory
import layer_factories.pooling_factories.f_globalmaxpooling3d_factory as f_globamaxpooling3d_factory
import layer_factories.pooling_factories.f_maxpooling1d_factory as f_maxpooling1d_factory
import layer_factories.pooling_factories.f_maxpooling2d_factory as f_maxpooling2d_factory
import layer_factories.pooling_factories.f_maxpooling3d_factory as f_maxpooling3d_factory

import layer_factories.recurrent_factories.f_basernn_factory as f_basernn_factory
import layer_factories.recurrent_factories.f_bidirectional_factory as f_bidirectional_factory
import layer_factories.recurrent_factories.f_convlstm1d_factory as f_convlstm1d_factory
import layer_factories.recurrent_factories.f_convlstm2d_factory as f_convlstm2d_factory
import layer_factories.recurrent_factories.f_convlstm3d_factory as f_convlstm3d_factory
import layer_factories.recurrent_factories.f_gru_factory as f_gru_factory
import layer_factories.recurrent_factories.f_grucell_factory as f_grucell_factory
import layer_factories.recurrent_factories.f_lstm_factory as f_lstm_factory
import layer_factories.recurrent_factories.f_lstmcell_factory as f_lstmcell_factory
import layer_factories.recurrent_factories.f_simplernn_factory as f_simplernn_factory
import layer_factories.recurrent_factories.f_simplernncell_factory as f_simplernncell_factory
import layer_factories.recurrent_factories.f_stackedrnncell_factory as f_stackedrnncell_factory
import layer_factories.recurrent_factories.f_timedistributed_factory as f_timedistributed_factory

import layer_factories.preprocessing_factories.f_autocontrast_factory as f_autocontrast_factory
import layer_factories.preprocessing_factories.f_categoryencoding_factory as f_categoryencoding_factory
import layer_factories.preprocessing_factories.f_centercrop_factory as f_centercrop_factory
import layer_factories.preprocessing_factories.f_discretization_factory as f_discretization_factory
import layer_factories.preprocessing_factories.f_hashedcrossing_factory as f_hashedcrossing_factory
import layer_factories.preprocessing_factories.f_hasing_factory as f_hasing_factory
import layer_factories.preprocessing_factories.f_integerlookup_factory as f_integerlookup_factory
import layer_factories.preprocessing_factories.f_melspectrogram_factory as f_melspectrogram_factory
import layer_factories.preprocessing_factories.f_normalization_factory as f_normalization_factory
import layer_factories.preprocessing_factories.f_pipeline_factory as f_pipeline_factory
import layer_factories.preprocessing_factories.f_randombrightness_factory as f_randombrightness_factory
import layer_factories.preprocessing_factories.f_randomcontrast_factory as f_randomcontrast_factory
import layer_factories.preprocessing_factories.f_randomcrop_factory as f_randomcrop_factory
import layer_factories.preprocessing_factories.f_randomflip_factory as f_randomflip_factory
import layer_factories.preprocessing_factories.f_randomrotation_factory as f_randomrotation_factory
import layer_factories.preprocessing_factories.f_randomtranslation_factory as f_randomtranslation_factory
import layer_factories.preprocessing_factories.f_randomzoom_factory as f_randomzoom_factory
import layer_factories.preprocessing_factories.f_rescaling_factory as f_rescaling_factory
import layer_factories.preprocessing_factories.f_resizing_factory as f_resizing_factory
import layer_factories.preprocessing_factories.f_solarization_factory as f_solarization_factory
import layer_factories.preprocessing_factories.f_spectralnormalization_factory as f_spectralnormalization_factory
import layer_factories.preprocessing_factories.f_stringlookup_factory as f_stringlookup_factory
import layer_factories.preprocessing_factories.f_textvectorization_factory as f_textvectorization_factory

import layer_factories.normalization_factories.f_batchnormalization_factory as f_batchnormalization_factory
import layer_factories.normalization_factories.f_groupnormalization_factory as f_groupnormalization_factory
import layer_factories.normalization_factories.f_layernormalization_factory as f_layernormalization_factory
import layer_factories.normalization_factories.f_unitnormalization_factory as f_unitnormalization_factory

import layer_factories.regularization_factories.f_activityregularization_factory as f_activityregularization_factory
import layer_factories.regularization_factories.f_alphadropout_factory as f_alphadropout_factory
import layer_factories.regularization_factories.f_dropout_factory as f_dropout_factory
import layer_factories.regularization_factories.f_gaussiandropout_factory as f_gaussiandropout_factory
import layer_factories.regularization_factories.f_gaussiannoise_factory as f_gaussiannoise_factory
import layer_factories.regularization_factories.f_spatialdropout1d_factory as f_spatialdropout1d_factory
import layer_factories.regularization_factories.f_spatialdropout2d_factory as f_spatialdropout2d_factory
import layer_factories.regularization_factories.f_spatialdropout3d_factory as f_spatialdropout3d_factory

def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "Activation":
            f_activation_factory.call(self, operation)
        case "Dense":
            f_dense_factory.call(self, operation)
        case "EinsumDense":
            f_einsumdense_factory.call(self, operation)
        case "Embedding":
            f_embedding_factory.call(self, operation)
        case "Identity":
            f_identity_factory.call(self, operation)
        case "Input":
            f_input_factory.call(self, operation)
        case "InputSpec":
            f_inputspec_factory.call(self, operation)
        case "Lambda":
            f_lambda_factory.call(self, operation)
        case "Masking":
            f_masking_factory.call(self, operation)

        case "Conv1D":
            f_conv1d_factory.call(self, operation)
        case "Conv1DTranspose":
            f_conv1dtranspose_factory.call(self, operation)
        case "Conv2D":
            f_conv2d_factory.call(self, operation)
        case "Conv2DTranspose":
            f_conv2dtranspose_factory.call(self, operation)
        case "Conv3D":
            f_conv3d_factory.call(self, operation)
        case "Conv3DTranspose":
            f_conv3dtranspose_factory.call(self, operation)
        case "DepthwiseConv1D":
            f_depthwiseconv1d_factory.call(self, operation)
        case "DepthwiseConv2D":
            f_depthwiseconv2d_factory.call(self, operation)
        case "SeparableConv1D":
            f_separableconv1d_factory.call(self, operation)
        case "SeparableConv2D":
            f_separableconv2d_factory.call(self, operation)

        case "AveragePooling1D":
            f_averagepooling1d_factory.call(self, operation)
        case "AveragePooling2D":
            f_averagepooling2d_factory.call(self, operation)
        case "AveragePooling3D":
            f_averagepooling3d_factory.call(self, operation)
        case "GlobalAveragePooling1D":
            f_globalaveragepooling1d_factory.call(self, operation)
        case "GlobalAveragePooling2D":
            f_globalaveragepooling2d_factory.call(self, operation)
        case "GlobalAveragePooling3D":
            f_globalaveragepooling3d_factory.call(self, operation)
        case "GlobalMaxPooling1D":
            f_globamaxpooling1d_factory.call(self, operation)
        case "GlobalMaxPooling2D":
            f_globamaxpooling2d_factory.call(self, operation)
        case "GlobalMaxPooling3D":
            f_globamaxpooling3d_factory.call(self, operation)
        case "MaxPooling1D":
            f_maxpooling1d_factory.call(self, operation)
        case "MaxPooling2D":
            f_maxpooling2d_factory.call(self, operation)
        case "MaxPooling3D":
            f_maxpooling3d_factory.call(self, operation)

        case "BaseRNN":
            f_basernn_factory.call(self, operation)
        case "Bidirectional":
            f_bidirectional_factory.call(self, operation)
        case "ConvLSTM1D":
            f_convlstm1d_factory.call(self, operation)
        case "ConvLSTM2D":
            f_convlstm2d_factory.call(self, operation)
        case "ConvLSTM3D":
            f_convlstm3d_factory.call(self, operation)
        case "GRU":
            f_gru_factory.call(self, operation)
        case "GRUCell":
            f_grucell_factory.call(self, operation)
        case "LSTM":
            f_lstm_factory.call(self, operation)
        case "LSTMCell":
            f_lstmcell_factory.call(self, operation)
        case "SimpleRNN":
            f_simplernn_factory.call(self, operation)
        case "SimpleRNNCell":
            f_simplernncell_factory.call(self, operation)
        case "StackedRNNCell":
            f_stackedrnncell_factory.call(self, operation)
        case "TimeDistributed":
            f_timedistributed_factory.call(self, operation)

        case "AutoContrast":
            f_autocontrast_factory.call(self, operation)
        case "CategoryEncoding":
            f_categoryencoding_factory.call(self, operation)
        case "CenterCrop":
            f_centercrop_factory.call(self, operation)
        case "Discretization":
            f_discretization_factory.call(self, operation)
        case "HashedCrossing":
            f_hashedcrossing_factory.call(self, operation)
        case "Hashing":
            f_hasing_factory.call(self, operation)
        case "IntegerLookup":
            f_integerlookup_factory.call(self, operation)
        case "MelSpectrogram":
            f_melspectrogram_factory.call(self, operation)
        case "Normalization":
            f_normalization_factory.call(self, operation)
        case "Pipeline":
            f_pipeline_factory.call(self, operation)
        case "RandomBrightness":
            f_randombrightness_factory.call(self, operation)
        case "RandomContrast":
            f_randomcontrast_factory.call(self, operation)
        case "RandomCrop":
            f_randomcrop_factory.call(self, operation)
        case "RandomFlip":
            f_randomflip_factory.call(self, operation)
        case "RandomRotation":
            f_randomrotation_factory.call(self, operation)
        case "RandomTranslation":
            f_randomtranslation_factory.call(self, operation)
        case "RandomZoom":
            f_randomzoom_factory.call(self, operation)
        case "Rescaling":
            f_rescaling_factory.call(self, operation)
        case "Resizing":
            f_resizing_factory.call(self, operation)
        case "Solarization":
            f_solarization_factory.call(self, operation)
        case "SpectralNormalization":
            f_spectralnormalization_factory.call(self, operation)
        case "StringLookup":
            f_stringlookup_factory.call(self, operation)
        case "TextVectorization":
            f_textvectorization_factory.call(self, operation)

        case "BatchNormalization":
            f_batchnormalization_factory.call(self, operation)
        case "GroupNormalization":
            f_groupnormalization_factory.call(self, operation)
        case "LayerNormalization":
            f_layernormalization_factory.call(self, operation)
        case "UnitNormalization":
            f_unitnormalization_factory.call(self, operation)

        case "ActivityRegularization":
            f_activityregularization_factory.call(self, operation)
        case "AlphaDropout":
            f_alphadropout_factory.call(self, operation)
        case "Dropout":
            f_dropout_factory.call(self, operation)
        case "GaussianDropout":
            f_gaussiandropout_factory.call(self, operation)
        case "GaussianNoise":
            f_gaussiannoise_factory.call(self, operation)
        case "SpatialDropout1D":
            f_spatialdropout1d_factory.call(self, operation)
        case "SpatialDropout2D":
            f_spatialdropout2d_factory.call(self, operation)
        case "SpatialDropout3D":
            f_spatialdropout3d_factory.call(self, operation)


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
