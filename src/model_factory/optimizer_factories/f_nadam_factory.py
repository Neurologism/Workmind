import tensorflow as tf
import keras


def call(self, operation: dict) -> None:
    self.project_data[operation["uid"]] = keras.optimizers.Nadam(
        learning_rate=(operation["args"]["learning_rate"] if "learning_rate" in operation["args"] else 0.001),
        beta_1=(operation["args"]["beta_1"] if "beta_1" in operation["args"] else 0.9),
        beta_2=(operation["args"]["beta_2"] if "beta_2" in operation["args"] else 0.999),
        epsilon=(operation["args"]["epsilon"] if "epsilon" in operation["args"] else 1e-07),
        weight_decay=(operation["args"]["weight_decay"] if "weight_decay" in operation["args"] else None),
        clipnorm=(operation["args"]["clipnorm"] if "clipnorm" in operation["args"] else None),
        clipvalue=(operation["args"]["clipvalue"] if "clipvalue" in operation["args"] else None),
        global_clipnorm=(operation["args"]["global_clipnorm"] if "global_clipnorm" in operation["args"] else None),
        use_ema=(operation["args"]["use_ema"] if "use_ema" in operation["args"] else False),
        ema_momentum=(operation["args"]["ema_momentum"] if "ema_momentum" in operation["args"] else 0.99),
        ema_overwrite_frequency=(operation["args"]["ema_overwrite_frequency"] if "ema_overwrite_frequency" in operation["args"] else None),
        loss_scale_factor=(operation["args"]["loss_scale_factor"] if "loss_scale_factor" in operation["args"] else None),
        gradient_accumulation_steps=(operation["args"]["gradient_accumulation_steps"] if "gradient_accumulation_steps" in operation["args"] else None),
        name=(operation["args"]["name"] if "name" in operation["args"] else "Nadam"),
    )