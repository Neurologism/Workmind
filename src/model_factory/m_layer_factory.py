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

def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
