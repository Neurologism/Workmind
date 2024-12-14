from src.tensorflow_middleware.m_dependencies import *


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.optimizers.LossScaleOptimizer(
        optimizer=(self.project_data[operation["args"]["optimizer"]]),
        loss_scale_factor=(
            operation["args"]["loss_scale_factor"]
            if "loss_scale_factor" in operation["args"]
            else 32768.0
        ),
        dynamic_growth_steps=(
            operation["args"]["dynamic_growth_steps"]
            if "dynamic_growth_steps" in operation["args"]
            else 2000
        ),
    )
