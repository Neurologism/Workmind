from .dependencies import *


def create_dataset(params: dict) -> tf.data.Dataset:
    params["as_supervised"] = True
    params["shuffle_files"] = True
    params["batch_size"] = 64

    params["name"] = params["datasetIdentifier"]

    load_params = {
        k: v for k, v in params.items() if k in tfds.load.__code__.co_varnames
    }

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
    "dataset": create_dataset,
}
