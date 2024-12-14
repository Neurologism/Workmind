from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.layers.Discretization(
        bin_boundaries=(
            operation["args"]["bin_boundaries"]
            if "bin_boundaries" in operation["args"]
            else None
        ),
        num_bins=(
            operation["args"]["num_bins"] if "num_bins" in operation["args"] else None
        ),
        epsilon=(
            operation["args"]["epsilon"] if "epsilon" in operation["args"] else 0.01
        ),
        output_mode=(
            operation["args"]["output_mode"]
            if "output_mode" in operation["args"]
            else "int"
        ),
        sparse=(
            operation["args"]["sparse"] if "sparse" in operation["args"] else False
        ),
        dtype=(operation["args"]["dtype"] if "dtype" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
