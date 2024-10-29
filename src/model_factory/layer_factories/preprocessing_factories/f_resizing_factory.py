import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Resizing(
        height=operation["args"]["height"],
        width=operation["args"]["width"],
        interpolation=(operation["args"]["interpolation"] if "interpolation" in operation["args"] else "bilinear"),
        crop_to_aspect_ratio=(operation["args"]["crop_to_aspect_ratio"] if "crop_to_aspect_ratio" in operation["args"] else False),
        pad_to_aspect_ratio=(operation["args"]["pad_to_aspect_ratio"] if "pad_to_aspect_ratio" in operation["args"] else False),
        fill_mode=(operation["args"]["fill_mode"] if "fill_mode" in operation["args"] else "constant"),
        fill_value=(operation["args"]["fill_value"] if "fill_value" in operation["args"] else 0.0),
        data_format=(operation["args"]["data_format"] if "data_format" in operation["args"] else None),
    )