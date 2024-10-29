import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.BatchNormalization(
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
        momentum=(operation["args"]["momentum"] if "momentum" in operation["args"] else 0.99),
        epsilon=(operation["args"]["epsilon"] if "epsilon" in operation["args"] else 0.001),
        center=(operation["args"]["center"] if "center" in operation["args"] else True),
        scale=(operation["args"]["scale"] if "scale" in operation["args"] else True),
        beta_initializer=(operation["args"]["beta_initializer"] if "beta_initializer" in operation["args"] else "zeros"),
        gamma_initializer=(operation["args"]["gamma_initializer"] if "gamma_initializer" in operation["args"] else "ones"),
        moving_mean_initializer=(operation["args"]["moving_mean_initializer"] if "moving_mean_initializer" in operation["args"] else "zeros"),
        moving_variance_initializer=(operation["args"]["moving_variance_initializer"] if "moving_variance_initializer" in operation["args"] else "ones"),
        beta_regularizer=(self.project_data[operation["args"]["beta_regularizer"]] if "beta_regularizer" in operation["args"] else None),
        gamma_regularizer=(self.project_data[operation["args"]["gamma_regularizer"]] if "gamma_regularizer" in operation["args"] else None),
        beta_constraint=(self.project_data[operation["args"]["beta_constraint"]] if "beta_constraint" in operation["args"] else None),
        gamma_constraint=(self.project_data[operation["args"]["gamma_constraint"]] if "gamma_constraint" in operation["args"] else None),
        synchronized=(operation["args"]["synchronized"] if "synchronized" in operation["args"] else False),
    )(self.project_data[operation["args"]["inputs"]])