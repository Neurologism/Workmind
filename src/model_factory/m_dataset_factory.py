import tensorflow as tf
import keras
import tensorflow_datasets as tfds


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        ds = tfds.load(
            operation["args"]["class"],
            split=operation["args"]["split"],
            shuffle_files=True,
            as_supervised=True,
        )
        ds = ds.batch(operation["args"]["batch_size"])
        assert isinstance(ds, tf.data.Dataset)
        if "normalize" in operation["args"]["preprocess"]:
            layer = keras.layers.Normalization()
            layer.adapt(ds.map(lambda x, y: x))
            ds = ds.map(lambda x, y: (layer(x), y))

        # expand with more preprocessing methods

        self.project_data[operation["uid"]] = ds
