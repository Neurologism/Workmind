from .optimizer_factories import f_adadelta_factory
from .optimizer_factories import f_adafactor_factory
from .optimizer_factories import f_adagrad_factory
from .optimizer_factories import f_adam_factory
from .optimizer_factories import f_adamax_factory
from .optimizer_factories import f_adamw_factory
from .optimizer_factories import f_ftrl_factory
from .optimizer_factories import f_lamb_factory
from .optimizer_factories import f_lion_factory
from .optimizer_factories import f_lossscaleoptimizer_factory
from .optimizer_factories import f_nadam_factory
from .optimizer_factories import f_rmsprop_factory
from .optimizer_factories import f_sgd_factory


def new(self, operation: dict) -> None:
    match operation["args"]["class"]:
        case "Adadelta":
            f_adadelta_factory.call(self, operation)
        case "Adafactor":
            f_adafactor_factory.call(self, operation)
        case "Adagrad":
            f_adagrad_factory.call(self, operation)
        case "Adam":
            f_adam_factory.call(self, operation)
        case "Adamax":
            f_adamax_factory.call(self, operation)
        case "AdamW":
            f_adamw_factory.call(self, operation)
        case "Ftrl":
            f_ftrl_factory.call(self, operation)
        case "Lamb":
            f_lamb_factory.call(self, operation)
        case "Lion":
            f_lion_factory.call(self, operation)
        case "LossScaleOptimizer":
            f_lossscaleoptimizer_factory.call(self, operation)
        case "Nadam":
            f_nadam_factory.call(self, operation)
        case "RMSprop":
            f_rmsprop_factory.call(self, operation)
        case "SGD":
            f_sgd_factory.call(self, operation)
        case _:
            raise ValueError(
                f"Optimizer class '{operation['args']['class']}' not recognized."
            )


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
