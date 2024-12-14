from ...m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Attention(
        use_scale=(
            operation["args"]["use_scale"]
            if "use_scale" in operation["args"]
            else False
        ),
        score_mode=(
            operation["args"]["score_mode"]
            if "score_mode" in operation["args"]
            else "dot"
        ),
        dropout=(
            operation["args"]["dropout"] if "dropout" in operation["args"] else 0.0
        ),
        seed=(operation["args"]["seed"] if "seed" in operation["args"] else None),
    )
