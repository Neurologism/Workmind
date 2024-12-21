class LineChart:
    def __init__(self, x_label: str, y_label: str, id: str) -> None:
        self.x_label = x_label
        self.y_label = y_label
        self.id = id

    def __call__(self, logs: dict, payload: dict) -> None:
        payload[self.id] = {
            "x": logs[self.x_label],
            "y": logs[self.y_label],
            "x_label": self.x_label,
            "y_label": self.y_label,
        }
