from f_dense_factory import call as f_dense_factory_call
from f_input_factory import call as f_input_factory_call


def new(self) -> None:
    if self.json_data["args"]["class"] == "Dense":
        f_dense_factory_call(self)

    if self.json_data["args"]["class"] == "Input":
        f_input_factory_call(self)


def call(self) -> None:
    if self.json_data["method"] == "new":
        new(self)
