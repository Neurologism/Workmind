import tensorflow as tf
import keras
from numpy.f2py.crackfortran import verbose


def create(self, operation: dict) -> None:
    self.project_data[operation["id"]] = keras.Model(
        self.project_data[operation["data"]["inputs"]],
        self.project_data[operation["data"]["outputs"]],
    )
    
def compile(self, operation: dict) -> None:
    self.project_data[operation["id"]].compile(
        optimizer=(
            operation["data"]["optimizer"]
            if "optimizer" in operation["data"]
            else "adam"
        ),
        loss=(
            operation["data"]["loss"]
            if "loss" in operation["data"]
            else None
        ),
        loss_weights=(
            operation["data"]["loss_weights"]
            if "loss_weights" in operation["data"]
            else None
        ),
        metrics=(
            [metric for metric in operation["data"]["metrics"]]
            if "metrics" in operation["data"]
            else None
        ),
        weighted_metrics=(
            operation["data"]["weighted_metrics"]
            if "weighted_metrics" in operation["data"]
            else None
        ),
        run_eagerly=(
            operation["data"]["run_eagerly"]
            if "run_eagerly" in operation["data"]
            else False
        ),
        steps_per_execution=(
            operation["data"]["steps_per_execution"]
            if "steps_per_execution" in operation["data"]
            else 1
        ),
        jit_compile=(
            operation["data"]["jit_compile"]
            if "jit_compile" in operation["data"]
            else "auto"
        ),
        auto_scale_loss=(
            operation["data"]["auto_scale_loss"]
            if "auto_scale_loss" in operation["data"]
            else True
        ),
    )
    
def fit(self, operation: dict) -> None:
    self.project_data[operation["id"]].fit(
        x=operation["data"]["dataset"],
        epochs=(
            operation["data"]["epochs"] if "epochs" in operation["data"] else 1
        ),
        verbose=(
            operation["data"]["verbose"]
            if "verbose" in operation["data"]
            else "auto"
        ),
        callbacks=(
            [
                self.project_data[callback]
                for callback in operation["data"]["callbacks"]
            ]
            if "callbacks" in operation["data"]
            else None
        ),
        validation_data=(
            operation["data"]["validation_data"]
            if "validation_data" in operation["data"]
            else None
        ),
        class_weight=(
            operation["data"]["class_weight"]
            if "class_weight" in operation["data"]
            else None
        ),
        sample_weight=(
            operation["data"]["sample_weight"]
            if "sample_weight" in operation["data"]
            else None
        ),
        initial_epoch=(
            operation["data"]["initial_epoch"]
            if "initial_epoch" in operation["data"]
            else 0
        ),
        steps_per_epoch=(
            operation["data"]["steps_per_epoch"]
            if "steps_per_epoch" in operation["data"]
            else None
        ),
        validation_steps=(
            operation["data"]["validation_steps"]
            if "validation_steps" in operation["data"]
            else None
        ),
        validation_freq=(
            operation["data"]["validation_freq"]
            if "validation_freq" in operation["data"]
            else 1
        ),
    )
    
def evaluate(self, operation: dict) -> None:
    self.project_data[operation["id"]].evaluate(
        x=operation["data"]["dataset"],
        verbose=(
            operation["data"]["verbose"]
            if "verbose" in operation["data"]
            else "auto"
        ),
        sample_weight=(
            operation["data"]["sample_weight"]
            if "sample_weight" in operation["data"]
            else None
        ),
        steps=(
            operation["data"]["steps"] if "steps" in operation["data"] else None
        ),
        callbacks=(
            [
                self.project_data[callback]
                for callback in operation["data"]["callbacks"]
            ]
            if "callbacks" in operation["data"]
            else None
        ),
        return_dict=(
            operation["data"]["return_dict"]
            if "return_dict" in operation["data"]
            else False
        ),
    )
    
def predict(self, operation: dict) -> None:
    self.project_data[operation["id"]].predict(
        x=self.project_data[operation["data"]["dataset"]],
        verbose=(
            operation["data"]["verbose"]
            if "verbose" in operation["data"]
            else "auto"
        ),
        steps=(
            operation["data"]["steps"] if "steps" in operation["data"] else None
        ),
        callbacks=(
            [
                self.project_data[callback]
                for callback in operation["data"]["callbacks"]
            ]
            if "callbacks" in operation["data"]
            else None
        ),
    )
    
def topo_sort(self, nodes: dict) -> list:
    visited = set()
    stack = []

    def dfs(node: dict) -> None:
        if node in visited:
            return
        visited.add(node)
        for child_id in node["data"]["out"]:
            if child_id in nodes:
                dfs(nodes[child_id])

        stack.append(node)

    for node in nodes.values():
        dfs(node)

    return stack[::-1]
    
def call(self, nodes: dict) -> None:
    sorted_nodes = topo_sort(self, nodes)

    for node in sorted_nodes:
        match node["identifier"]:
            case "create":
                create(self, node)

            case "compile":
                compile(self, node)

            case "fit":
                fit(self, node)

            case "evaluate":
                evaluate(self, node)

            case "predict":
                predict(self, node)
