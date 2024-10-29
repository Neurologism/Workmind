import optimizer_factories.f_adadelta_factory as f_adadelta_factory
import optimizer_factories.f_adafactor_factory as f_adafactor_factory
import optimizer_factories.f_adagrad_factory as f_adagrad_factory
import optimizer_factories.f_adam_factory as f_adam_factory
import optimizer_factories.f_adamax_factory as f_adamax_factory
import optimizer_factories.f_adamw_factory as f_adamw_factory
import optimizer_factories.f_ftrl_factory as f_ftrl_factory
import optimizer_factories.f_lamb_factory as f_lamb_factory
import optimizer_factories.f_lion_factory as f_lion_factory
import optimizer_factories.f_lossscaleoptimizer_factory as f_lossscaleoptimizer_factory
import optimizer_factories.f_nadam_factory as f_nadam_factory
import optimizer_factories.f_rmsprop_factory as f_rmsprop_factory
import optimizer_factories.f_sgd_factory as f_sgd_factory

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
            raise ValueError(f"Optimizer class '{operation['args']['class']}' not recognized.")

def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)