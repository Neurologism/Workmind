import tensorflow as tf
import keras
import tensorflow_datasets as tfds



ds = tfds.load(
            "mnist",
            split="train",
            shuffle_files=True,
            as_supervised=True,
        )
ds = ds.batch(32)

layer = keras.layers.Normalization()
layer.adapt(ds.map(lambda x, y: x))
ds = ds.map(lambda x, y: (layer(x), y))

input = keras.layers.Input(shape=(28,28))
x = keras.layers.Flatten()(input)
x = keras.layers.Dense(128, activation="relu")(x)
output = keras.layers.Dense(10, activation="softmax")(x)

model = keras.models.Model(inputs=input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(ds, epochs=1)



