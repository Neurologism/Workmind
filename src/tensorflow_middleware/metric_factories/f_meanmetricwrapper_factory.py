from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.metrics.MeanMetricWrapper(
        fn=self.project_data[operation["args"]["fn"]],
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
    )
