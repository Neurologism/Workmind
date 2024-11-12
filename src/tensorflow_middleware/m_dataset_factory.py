import tensorflow as tf
import keras
import tensorflow_datasets as tfds


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        ds = tfds.load(
            name=operation["args"]["name"],
            split=(
                operation["args"]["split"] if "split" in operation["args"] else None
            ),
            data_dir=(
                operation["args"]["data_dir"]
                if "data_dir" in operation["args"]
                else None
            ),
            batch_size=(
                operation["args"]["batch_size"]
                if "batch_size" in operation["args"]
                else None
            ),
            shuffle_files=(
                operation["args"]["shuffle_files"]
                if "shuffle_files" in operation["args"]
                else True
            ),
            download=(
                operation["args"]["download"]
                if "download" in operation["args"]
                else True
            ),
            as_supervised=(
                operation["args"]["as_supervised"]
                if "as_supervised" in operation["args"]
                else True
            ),
            decoders=(
                [
                    self.project_data[decoder]
                    for decoder in operation["args"]["decoders"]
                ]
                if "decoders" in operation["args"]
                else None
            ),
            read_config=(
                self.project_data[operation["args"]["read_config"]]
                if "read_config" in operation["args"]
                else None
            ),
            with_info=(
                operation["args"]["with_info"]
                if "with_info" in operation["args"]
                else False
            ),
            builder_kwargs=(
                operation["args"]["builder_kwargs"]
                if "builder_kwargs" in operation["args"]
                else None
            ),
            download_and_prepare_kwargs=(
                operation["args"]["download_and_prepare_kwargs"]
                if "download_and_prepare_kwargs" in operation["args"]
                else None
            ),
            as_dataset_kwargs=(
                operation["args"]["as_dataset_kwargs"]
                if "as_dataset_kwargs" in operation["args"]
                else None
            ),
            try_gcs=(
                operation["args"]["try_gcs"]
                if "try_gcs" in operation["args"]
                else False
            ),
        )
        assert isinstance(ds, tf.data.Dataset)
        if "normalize" in operation["args"]["preprocess"]:
            layer = keras.layers.Normalization()
            layer.adapt(ds.map(lambda x, y: x))
            ds = ds.map(lambda x, y: (layer(x), y))

        # expand with more preprocessing methods

        self.project_data[operation["uid"]] = ds
