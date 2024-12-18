from .visualizer_factories.f_line_chart_factory import LineChart


def call(self, nodes: dict) -> None:
    self.project_data["visualizer"] = []
    for node in nodes.values():
        if "type" in node:
            match node["type"]:
                case "line-chart":
                    self.project_data["visualizer"].append(
                        LineChart(
                            x_label=node["data"]["x_label"],
                            y_label=node["data"]["y_label"],
                            id=node["id"],
                        )
                    )
