from ..m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.Hinge(
        name=(operation["args"]["name"] if "name" in operation["args"] else "hinge"),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
