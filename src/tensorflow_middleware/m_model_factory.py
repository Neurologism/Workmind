import tensorflow as tf
import keras
from mkl_random.mklrand import shuffle
from numpy.f2py.crackfortran import verbose


def call(self, operation: dict) -> None:
    if operation["method"] == "new":
        self.project_data[operation["uid"]] = keras.Model(
            self.project_data[operation["args"]["inputs"]],
            self.project_data[operation["args"]["outputs"]],
        )

    elif operation["method"] == "compile":
        self.project_data[operation["uid"]].compile(
            optimizer=(self.project_data[operation["args"]["optimizer"]] if "optimizer" in operation["args"] else "adam"),
            loss=(self.project_data[operation["args"]["loss"]] if "loss" in operation["args"] else None),
            loss_weights=(operation["args"]["loss_weights"] if "loss_weights" in operation["args"] else None),
            metrics=([self.project_data[metric] for metric in operation["args"]["metrics"]] if "metrics" in operation["args"] else None),
            weighted_metrics=(operation["args"]["weighted_metrics"] if "weighted_metrics" in operation["args"] else None),
            run_eagerly=(operation["args"]["run_eagerly"] if "run_eagerly" in operation["args"] else False),
            steps_per_execution=(operation["args"]["steps_per_execution"] if "steps_per_execution" in operation["args"] else 1),
            jit_compile=(operation["args"]["jit_compile"] if "jit_compile" in operation["args"] else "auto"),
            auto_scale_loss=(operation["args"]["auto_scale_loss"] if "auto_scale_loss" in operation["args"] else True),
        )

    elif operation["method"] == "fit":
        self.project_data[operation["uid"]].fit(
            x=self.project_data[operation["args"]["dataset"]],
            epochs=(operation["args"]["epochs"] if "epochs" in operation["args"] else 1),
            verbose=(operation["args"]["verbose"] if "verbose" in operation["args"] else "auto"),
            callbacks=([self.project_data[callback] for callback in operation["args"]["callbacks"]] if "callbacks" in operation["args"] else None),
            validation_data=(self.project_data[operation["args"]["validation_data"]] if "validation_data" in operation["args"] else None),
            class_weight=(operation["args"]["class_weight"] if "class_weight" in operation["args"] else None),
            sample_weight=(operation["args"]["sample_weight"] if "sample_weight" in operation["args"] else None),
            initial_epoch=(operation["args"]["initial_epoch"] if "initial_epoch" in operation["args"] else 0),
            steps_per_epoch=(operation["args"]["steps_per_epoch"] if "steps_per_epoch" in operation["args"] else None),
            validation_steps=(operation["args"]["validation_steps"] if "validation_steps" in operation["args"] else None),
            validation_freq=(operation["args"]["validation_freq"] if "validation_freq" in operation["args"] else 1),
        )

    elif operation["method"] == "evaluate":
        self.project_data[operation["uid"]].evaluate(
            x=self.project_data[operation["args"]["dataset"]],
            verbose=(operation["args"]["verbose"] if "verbose" in operation["args"] else "auto"),
            sample_weight=(operation["args"]["sample_weight"] if "sample_weight" in operation["args"] else None),
            steps=(operation["args"]["steps"] if "steps" in operation["args"] else None),
            callbacks=([self.project_data[callback] for callback in operation["args"]["callbacks"]] if "callbacks" in operation["args"] else None),
            return_dict=(operation["args"]["return_dict"] if "return_dict" in operation["args"] else False),
        )

    elif operation["method"] == "predict":
        self.project_data[operation["uid"]].predict(
            x=self.project_data[operation["args"]["dataset"]],
            verbose=(operation["args"]["verbose"] if "verbose" in operation["args"] else "auto"),
            steps=(operation["args"]["steps"] if "steps" in operation["args"] else None),
            callbacks=([self.project_data[callback] for callback in operation["args"]["callbacks"]] if "callbacks" in operation["args"] else None),
        )