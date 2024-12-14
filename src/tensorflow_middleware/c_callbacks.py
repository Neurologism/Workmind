from .m_dependencies import *

class DatabaseLogger(Callback):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def on_train_begin(self, logs=None):
        payload = {
            "event": "train_start",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
            "logs": logs or {},
        }
        self.log(payload)

    def on_epoch_begin(self, epoch, logs=None):
        payload = {
            "event": "epoch_begin",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
            "epoch": epoch,
            "performance": logs or {},
        }
        self.log(payload)

    def on_epoch_end(self, epoch, logs=None):
        payload = {
            "event": "epoch_end",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
            "epoch": epoch,
            "performance": logs or {},
        }
        self.log(payload)

    def on_train_end(self, logs=None):
        payload = {
            "event": "train_end",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
                :-3
            ]
            + "Z",
            "logs": logs or {},
        }
        self.log(payload)
