from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.RandomZoom(
        height_factor=operation["args"]["height_factor"],
        width_factor=(
            operation["args"]["width_factor"]
            if "width_factor" in operation["args"]
            else None
        ),
        fill_mode=(
            operation["args"]["fill_mode"]
            if "fill_mode" in operation["args"]
            else "reflect"
        ),
        interpolation=(
            operation["args"]["interpolation"]
            if "interpolation" in operation["args"]
            else "bilinear"
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        fill_value=(
            operation["args"]["fill_value"]
            if "fill_value" in operation["args"]
            else 0.0
        ),
        data_format=(
            operation["args"]["data_format"]
            if "data_format" in operation["args"]
            else None
        ),
    )
