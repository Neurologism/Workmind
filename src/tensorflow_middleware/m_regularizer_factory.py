import regularizer_factories.f_l1_factory as f_l1_factory
import regularizer_factories.f_l1l2_factory as f_l1l2_factory
import regularizer_factories.f_l2_factory as f_l2_factory


def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "L1":
            f_l1_factory.call(self, operation)
        case "L1L2":
            f_l1l2_factory.call(self, operation)
        case "L2":
            f_l2_factory.call(self, operation)
        case _:
            raise ValueError(
                f"Regularizer class '{operation['args']['class']}' not recognized."
            )


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
