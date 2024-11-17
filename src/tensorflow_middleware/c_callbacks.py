from keras import callbacks
from datetime import datetime


class DatabaseLogger(callbacks.Callback):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def on_train_begin(self, logs=None):
        payload = {
            "event": "train_start",
            "timestamp": datetime.now().timestamp(),
            "logs": logs or {},
        }
        self.log(payload)

    def on_epoch_begin(self, epoch, logs=None):
        payload = {
            "event": "epoch_begin",
            "timestamp": datetime.now().timestamp(),
            "epoch": epoch,
            "performance": logs or {},
        }
        self.log(payload)

    def on_epoch_end(self, epoch, logs=None):
        payload = {
            "event": "epoch_end",
            "timestamp": datetime.now().timestamp(),
            "epoch": epoch,
            "performance": logs or {},
        }
        self.log(payload)

    def on_train_end(self, logs=None):
        payload = {
            "event": "train_end",
            "timestamp": datetime.now().timestamp(),
            "logs": logs or {},
        }
        self.log(payload)
