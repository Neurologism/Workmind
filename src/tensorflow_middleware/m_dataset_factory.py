import tensorflow as tf
import keras
import tensorflow_datasets as tfds


def call(self, operation: dict) -> None:
    if operation["identifier"] == "load":
        ds = tfds.load(
            name=operation["data"]["name"],
            split=(
                operation["data"]["split"] if "split" in operation["data"] else None
            ),
            data_dir=(
                operation["data"]["data_dir"]
                if "data_dir" in operation["data"]
                else None
            ),
            batch_size=(
                operation["data"]["batch_size"]
                if "batch_size" in operation["data"]
                else None
            ),
            shuffle_files=(
                operation["data"]["shuffle_files"]
                if "shuffle_files" in operation["data"]
                else True
            ),
            download=(
                operation["data"]["download"]
                if "download" in operation["data"]
                else True
            ),
            as_supervised=(
                operation["data"]["as_supervised"]
                if "as_supervised" in operation["data"]
                else True
            ),
            decoders=(
                [
                    self.project_data[decoder]
                    for decoder in operation["data"]["decoders"]
                ]
                if "decoders" in operation["data"]
                else None
            ),
            read_config=(
                self.project_data[operation["data"]["read_config"]]
                if "read_config" in operation["data"]
                else None
            ),
            with_info=(
                operation["data"]["with_info"]
                if "with_info" in operation["data"]
                else False
            ),
            builder_kwargs=(
                operation["data"]["builder_kwargs"]
                if "builder_kwargs" in operation["data"]
                else None
            ),
            download_and_prepare_kwargs=(
                operation["data"]["download_and_prepare_kwargs"]
                if "download_and_prepare_kwargs" in operation["data"]
                else None
            ),
            as_dataset_kwargs=(
                operation["data"]["as_dataset_kwargs"]
                if "as_dataset_kwargs" in operation["data"]
                else None
            ),
            try_gcs=(
                operation["data"]["try_gcs"]
                if "try_gcs" in operation["data"]
                else False
            ),
        )
        # assert isinstance(ds, tf.data.Dataset)
        # if "normalize" in operation["data"]["preprocess"]:
        #     layer = keras.layers.Normalization()
        #     layer.adapt(ds.map(lambda x, y: x))
        #     ds = ds.map(lambda x, y: (layer(x), y))

        # expand with more preprocessing methods

        self.project_data[operation["id"]] = ds
