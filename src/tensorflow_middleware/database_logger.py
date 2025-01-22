from .dependencies import *


class DatabaseLogger(Callback):
    def __init__(self, log) -> None:
        super().__init__()
        self.log = log
        self.block_payloads = {}

    def on_train_begin(self, logs=None):
        if self.log is None:
            return
        payload = {
            "event": "train_start",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
        }
        self.log(payload)

    def on_epoch_begin(self, epoch, logs=None):
        pass
        # payload = {
        #     "event": "epoch_begin",
        #     "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        #         :-3
        #     ]
        #     + "Z",
        #     "epoch": epoch,
        #     # "performance": logs or {},
        # }
        # self.log(payload)

    def on_epoch_end(self, epoch, logs=None):
        if self.log is None:
            return
        payload = {
            "event": "epoch_end",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
            "epoch": epoch,
            "performance": logs or {},
        }
        logs["epoch"] = epoch
        for block_id, responses in self.block_payloads:
            payload[block_id] = {}

            for key, value in responses[0].items():
                if value in logs:
                    payload[block_id][key] = logs[value]

            if responses.length > 1:
                payload[block_id].update(responses[1])

        self.log(payload)

    def on_train_end(self, logs=None):
        if self.log is None:
            return
        payload = {
            "event": "train_end",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
            "logs": logs or {},
        }
        self.log(payload)
