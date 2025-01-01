import keras.src.callbacks

from .m_dependencies import *


def create(self, operation: dict) -> None:
    self.project_data[operation["data"]["name"]] = keras.Model(
        self.project_data[operation["data"]["input"][0][0]],
        self.project_data[operation["data"]["output"][0][0]],
    )


def compile(self, operation: dict) -> None:
    self.project_data[operation["data"]["name"]].compile(
        optimizer=(
            operation["data"]["optimizer"]
            if "optimizer" in operation["data"]
            else "adam"
        ),
        loss=(operation["data"]["loss"] if "loss" in operation["data"] else None),
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
    self.project_data[operation["data"]["name"]].fit(
        x=self.project_data[operation["data"]["data"][0][0]][
            operation["data"]["data"][0][1]
        ],
        epochs=(operation["data"]["epochs"] if "epochs" in operation["data"] else 1),
        verbose=(
            operation["data"]["verbose"] if "verbose" in operation["data"] else "auto"
        ),
        callbacks=(
            (
                [
                    self.project_data[callback]
                    for callback in operation["data"]["callbacks"]
                ]
                if "callbacks" in operation["data"]
                else []
            )
            + self.callbacks
            + (
                [
                    keras.callbacks.EarlyStopping(
                        patience=operation["data"]["early_stopping"],
                        restore_best_weights=True,
                    )
                ]
                if operation["data"]["early_stopping"] is not None
                else []
            )
        ),
        validation_data=(
            self.project_data[operation["data"]["validation_data"][0][0]][
                operation["data"]["validation_data"][0][1]
            ]
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
    self.project_data[operation["data"]["name"]].evaluate(
        x=self.project_data[operation["data"]["data"][0][0]][
            operation["data"]["data"][0][1]
        ],
        verbose=(
            operation["data"]["verbose"] if "verbose" in operation["data"] else "auto"
        ),
        sample_weight=(
            operation["data"]["sample_weight"]
            if "sample_weight" in operation["data"]
            else None
        ),
        steps=(operation["data"]["steps"] if "steps" in operation["data"] else None),
        callbacks=(
            [self.project_data[callback] for callback in operation["data"]["callbacks"]]
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
    self.project_data[operation["data"]["name"]].predict(
        x=self.project_data[operation["data"]["data"][0][0]][
            operation["data"]["data"][0][1]
        ],
        verbose=(
            operation["data"]["verbose"] if "verbose" in operation["data"] else "auto"
        ),
        steps=(operation["data"]["steps"] if "steps" in operation["data"] else None),
        callbacks=(
            [self.project_data[callback] for callback in operation["data"]["callbacks"]]
            if "callbacks" in operation["data"]
            else None
        ),
    )


def topo_sort(self, nodes: dict) -> list:
    visited = set()
    stack = []

    def dfs(node: dict) -> None:
        if node["id"] in visited:
            return
        visited.add(node["id"])
        if "out" in node["data"]:
            for child_id in node["data"]["out"]:
                if child_id[0] in nodes:
                    dfs(nodes[child_id[0]])

        stack.append(node)

    for node in nodes.values():
        dfs(node)

    return stack[::-1]


def call(self, nodes: dict) -> None:
    sorted_nodes = topo_sort(self, nodes)

    for node in sorted_nodes:
        match node["identifier"]:
            case "Model":
                if "dataset" in node["data"]:
                    create(self, node)
                    compile(self, node)

            case "fit":
                if node["data"]["name"] in self.project_data:
                    fit(self, node)

            case "evaluate":
                if node["data"]["name"] in self.project_data:
                    evaluate(self, node)

            case "predict":
                if node["data"]["name"] in self.project_data:
                    predict(self, node)

            case _:
                raise ValueError(f"Unknown operation: {node['identifier']}")
