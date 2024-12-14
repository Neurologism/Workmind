from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.SpatialDropout1D(
        rate=operation["args"]["rate"],
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
