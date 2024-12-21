from ...m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.StackedRNNCells(
        cells=[self.project_data[cell] for cell in operation["args"]["cells"]],
    )
