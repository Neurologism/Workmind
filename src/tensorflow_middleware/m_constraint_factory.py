from .constraint_factories import f_maxnorm_factory
from .constraint_factories import f_minmaxnorm_factory
from .constraint_factories import f_nonneg_factory
from .constraint_factories import f_unitnorm_factory


def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "MaxNorm":
            f_maxnorm_factory.call(self, operation)
        case "MinMaxNorm":
            f_minmaxnorm_factory.call(self, operation)
        case "NonNeg":
            f_nonneg_factory.call(self, operation)
        case "UnitNorm":
            f_unitnorm_factory.call(self, operation)
        case _:
            raise ValueError(
                f"Constraint class '{operation['args']['class']}' not recognized."
            )


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
