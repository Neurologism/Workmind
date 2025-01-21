from .dependencies import *


def create_dataset(params: dict) -> tf.data.Dataset:
    load_params = inspect.signature(tfds.load).parameters
    load_params = {k: v for k, v in params.items() if k in load_params}

    dataset = tfds.load(**load_params)

    if params["name"] == "wine_quality":

        def preprocess(features, label):
            feature_list = [
                tf.cast(features[key], np.float32) for key in sorted(features.keys())
            ]
            return tf.stack(feature_list, axis=-1), label

        dataset["train"] = dataset["train"].map(preprocess)

    if isinstance(dataset, tf.data.Dataset):
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        for key in dataset:
            dataset[key] = dataset[key].prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


dataset_to_function = {
    "create": create_dataset,
}
