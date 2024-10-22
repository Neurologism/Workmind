import layer_factories.f_dense_factory as f_dense_factory
import layer_factories.f_input_factory as f_input_factory


def new(self, operation: dict) -> None:
    if operation["args"]["class"] == "Dense":
        f_dense_factory.call(self, operation)

    if operation["args"]["class"] == "Input":
        f_input_factory.call(self, operation)


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        new(self, operation)
