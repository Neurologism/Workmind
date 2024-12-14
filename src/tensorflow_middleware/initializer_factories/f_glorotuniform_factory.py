from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.initializers.GlorotUniform(
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
