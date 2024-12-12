import tensorflow as tf
import keras
import tensorflow_datasets as tfds


def load(self, operation: dict) -> None:
    ds = tfds.load(
        name=operation["type"],
        split=(operation["data"]["split"] if "split" in operation["data"] else None),
        data_dir=(
            operation["data"]["data_dir"] if "data_dir" in operation["data"] else None
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
            operation["data"]["download"] if "download" in operation["data"] else True
        ),
        as_supervised=(
            operation["data"]["as_supervised"]
            if "as_supervised" in operation["data"]
            else True
        ),
        decoders=(
            [self.project_data[decoder] for decoder in operation["data"]["decoders"]]
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
            operation["data"]["try_gcs"] if "try_gcs" in operation["data"] else False
        ),
    )
    if isinstance(ds, tf.data.Dataset):
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        for key in ds:
            ds[key] = ds[key].prefetch(tf.data.experimental.AUTOTUNE)
    self.project_data[operation["id"]] = ds


def split(self, operation: dict) -> None:
    result = {}
    dataset = self.project_data[operation["data"]["in"][0][0]][
        operation["data"]["in"][0][1]
    ]

    split_ratio = operation["data"]["ratio"]
    total_size = dataset.cardinality().numpy()
    split_index = int(total_size * split_ratio)

    result["split1"] = dataset.take(split_index)
    result["split2"] = dataset.skip(split_index)

    result["split1"] = result["split1"].prefetch(tf.data.experimental.AUTOTUNE)
    result["split2"] = result["split2"].prefetch(tf.data.experimental.AUTOTUNE)

    self.project_data[operation["id"]] = result


def call(self, nodes: dict) -> None:
    for node in nodes.values():
        if node["type"] != "split":
            load(self, node)

    for node in nodes.values():
        if node["type"] == "split":
            split(self, node)
