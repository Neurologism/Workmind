import tensorflow as tf
import keras
import tensorflow_datasets as tfds

def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        ds = tfds.load(
            operation["args"]["class"],
            operation["args"]["split"],
            shuffle_files=True,
        )
        if "normalization" in operation["args"]["preprocess"]:
            layer = keras.layers.Normalization()
            layer.adapt(ds)
            ds = layer(ds)

        # expand with more preprocessing methods

        self.project_data[operation["uid"]] = ds


