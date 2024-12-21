from ..m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.regularizers.L1L2(
        l1=(operation["args"]["l1"] if "l1" in operation["args"] else 0.01),
        l2=(operation["args"]["l2"] if "l2" in operation["args"] else 0.01),
    )
