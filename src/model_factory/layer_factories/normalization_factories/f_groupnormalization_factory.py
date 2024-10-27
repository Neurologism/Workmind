import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.GroupNormalization(
        groups=(operation["args"]["groups"] if "groups" in operation["args"] else 32),
        axis=(operation["args"]["axis"] if "axis" in operation["args"] else -1),
        epsilon=(operation["args"]["epsilon"] if "epsilon" in operation["args"] else 1e-3),
        center=(operation["args"]["center"] if "center" in operation["args"] else True),
        scale=(operation["args"]["scale"] if "scale" in operation["args"] else True),
        beta_initializer=(operation["args"]["beta_initializer"] if "beta_initializer" in operation["args"] else "zeros"),
        gamma_initializer=(operation["args"]["gamma_initializer"] if "gamma_initializer" in operation["args"] else "ones"),
    )(self.project_data[operation["args"]["inputs"]])