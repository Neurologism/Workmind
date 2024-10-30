import constraint_factories.f_maxnorm_factory as f_maxnorm_factory
import constraint_factories.f_minmaxnorm_factory as f_minmaxnorm_factory
import constraint_factories.f_nonneg_factory as f_nonneg_factory
import constraint_factories.f_unitnorm_factory as f_unitnorm_factory


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
