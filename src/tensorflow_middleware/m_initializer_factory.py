from .initializer_factories import f_constant_factory
from .initializer_factories import f_glorotnormal_factory
from .initializer_factories import f_glorotuniform_factory
from .initializer_factories import f_henormal_factory
from .initializer_factories import f_heuniform_factory
from .initializer_factories import f_identity_factory
from .initializer_factories import f_lecunnormal_factory
from .initializer_factories import f_lecununiform_factory
from .initializer_factories import f_ones_factory
from .initializer_factories import f_orthogonal_factory
from .initializer_factories import f_randomnormal_factory
from .initializer_factories import f_randomuniform_factory
from .initializer_factories import f_truncatednormal_factory
from .initializer_factories import f_variance_scaling_factory
from .initializer_factories import f_zeros_factory


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
            raise ValueError(
                f"Initializer class '{operation['args']['class']}' not recognized."
            )


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
