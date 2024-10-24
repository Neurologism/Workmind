import tensorflow as tf
import keras
from keras.src.utils.module_utils import tensorflow


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Input(
        shape=tuple(x for x in operation["args"]["shape"]),
    )
