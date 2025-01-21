def line_chart(params: dict):
    params["logger"][params["id"]] = [
        {
            "x": params["x_label"],
            "y": params["y_label"],
            "val_y": "val_" + params["y_label"],
        },
        {
            "x_label": params["x_label"],
            "y_label": params["y_label"],
        },
    ]


visualizer_to_function = {
    "line-chart": line_chart,
}
