from ..m_dependencies import *

def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.optimizers.RMSprop(
        learning_rate=(
            operation["args"]["learning_rate"]
            if "learning_rate" in operation["args"]
            else 0.001
        ),
        rho=(operation["args"]["rho"] if "rho" in operation["args"] else 0.9),
        momentum=(
            operation["args"]["momentum"] if "momentum" in operation["args"] else 0.0
        ),
        epsilon=(
            operation["args"]["epsilon"] if "epsilon" in operation["args"] else 1e-07
        ),
        centered=(
            operation["args"]["centered"] if "centered" in operation["args"] else False
        ),
        weight_decay=(
            operation["args"]["weight_decay"]
            if "weight_decay" in operation["args"]
            else None
        ),
        clipnorm=(
            operation["args"]["clipnorm"] if "clipnorm" in operation["args"] else None
        ),
        clipvalue=(
            operation["args"]["clipvalue"] if "clipvalue" in operation["args"] else None
        ),
        global_clipnorm=(
            operation["args"]["global_clipnorm"]
            if "global_clipnorm" in operation["args"]
            else None
        ),
        use_ema=(
            operation["args"]["use_ema"] if "use_ema" in operation["args"] else False
        ),
        ema_momentum=(
            operation["args"]["ema_momentum"]
            if "ema_momentum" in operation["args"]
            else 0.99
        ),
        ema_overwrite_frequency=(
            operation["args"]["ema_overwrite_frequency"]
            if "ema_overwrite_frequency" in operation["args"]
            else None
        ),
        loss_scale_factor=(
            operation["args"]["loss_scale_factor"]
            if "loss_scale_factor" in operation["args"]
            else None
        ),
        gradient_accumulation_steps=(
            operation["args"]["gradient_accumulation_steps"]
            if "gradient_accumulation_steps" in operation["args"]
            else None
        ),
        name=(operation["args"]["name"] if "name" in operation["args"] else None),
    )
