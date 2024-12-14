from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.layers.BatchNormalization(
        axis=(operation["data"]["axis"] if "axis" in operation["data"] else -1),
        momentum=(
            operation["data"]["momentum"] if "momentum" in operation["data"] else 0.99
        ),
        epsilon=(
            operation["data"]["epsilon"] if "epsilon" in operation["data"] else 0.001
        ),
        center=(operation["data"]["center"] if "center" in operation["data"] else True),
        scale=(operation["data"]["scale"] if "scale" in operation["data"] else True),
        beta_initializer=(
            operation["data"]["beta_initializer"]
            if "beta_initializer" in operation["data"]
            else "zeros"
        ),
        gamma_initializer=(
            operation["data"]["gamma_initializer"]
            if "gamma_initializer" in operation["data"]
            else "ones"
        ),
        moving_mean_initializer=(
            operation["data"]["moving_mean_initializer"]
            if "moving_mean_initializer" in operation["data"]
            else "zeros"
        ),
        moving_variance_initializer=(
            operation["data"]["moving_variance_initializer"]
            if "moving_variance_initializer" in operation["data"]
            else "ones"
        ),
        beta_regularizer=(
            self.project_data[operation["data"]["beta_regularizer"]]
            if "beta_regularizer" in operation["data"]
            else None
        ),
        gamma_regularizer=(
            self.project_data[operation["data"]["gamma_regularizer"]]
            if "gamma_regularizer" in operation["data"]
            else None
        ),
        beta_constraint=(
            self.project_data[operation["data"]["beta_constraint"]]
            if "beta_constraint" in operation["data"]
            else None
        ),
        gamma_constraint=(
            self.project_data[operation["data"]["gamma_constraint"]]
            if "gamma_constraint" in operation["data"]
            else None
        ),
        synchronized=(
            operation["data"]["synchronized"]
            if "synchronized" in operation["data"]
            else False
        ),
    )(self.project_data[operation["data"]["inputs"]])
