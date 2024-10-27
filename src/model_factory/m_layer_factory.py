import src.model_factory.layer_factories.core_factories.f_dense_factory as f_dense_factory
import src.model_factory.layer_factories.core_factories.f_input_factory as f_input_factory
import src.model_factory.layer_factories.core_factories.f_flatten_factory as f_flatten_factory


def new(self, operation: dict) -> None:
    if operation["args"]["class"] == "Dense":
        f_dense_factory.call(self, operation)

    elif operation["args"]["class"] == "Input":
        f_input_factory.call(self, operation)

    elif operation["args"]["class"] == "Flatten":
        f_flatten_factory.call(self, operation)


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
