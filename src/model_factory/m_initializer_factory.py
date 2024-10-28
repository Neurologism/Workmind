import initializer_factories.f_constant_factory as f_constant_factory
import initializer_factories.f_glorotnormal_factory as f_glorotnormal_factory
import initializer_factories.f_glorotuniform_factory as f_glorotuniform_factory
import initializer_factories.f_henormal_factory as f_henormal_factory
import initializer_factories.f_heuniform_factory as f_heuniform_factory
import initializer_factories.f_identity_factory as f_identity_factory
import initializer_factories.f_lecunnormal_factory as f_lecunnormal_factory
import initializer_factories.f_lecununiform_factory as f_lecununiform_factory
import initializer_factories.f_ones_factory as f_ones_factory
import initializer_factories.f_orthogonal_factory as f_orthogonal_factory
import initializer_factories.f_randomnormal_factory as f_randomnormal_factory
import initializer_factories.f_randomuniform_factory as f_randomuniform_factory
import initializer_factories.f_truncatednormal_factory as f_truncatednormal_factory
import initializer_factories.f_variancescaling_factory as f_variance_scaling_factory
import initializer_factories.f_zeros_factory as f_zeros_factory


def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "Constant":
            f_constant_factory.call(self, operation)
        case "GlorotNormal":
            f_glorotnormal_factory.call(self, operation)
        case "GlorotUniform":
            f_glorotuniform_factory.call(self, operation)
        case "HeNormal":
            f_henormal_factory.call(self, operation)
        case "HeUniform":
            f_heuniform_factory.call(self, operation)
        case "Identity":
            f_identity_factory.call(self, operation)
        case "LecunNormal":
            f_lecunnormal_factory.call(self, operation)
        case "LecunUniform":
            f_lecununiform_factory.call(self, operation)
        case "Ones":
            f_ones_factory.call(self, operation)
        case "Orthogonal":
            f_orthogonal_factory.call(self, operation)
        case "RandomNormal":
            f_randomnormal_factory.call(self, operation)
        case "RandomUniform":
            f_randomuniform_factory.call(self, operation)
        case "TruncatedNormal":
            f_truncatednormal_factory.call(self, operation)
        case "VarianceScaling":
            f_variance_scaling_factory.call(self, operation)
        case "Zeros":
            f_zeros_factory.call(self, operation)
        case _:
            raise ValueError(f"Initializer class '{operation['args']['class']}' not recognized.")


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)