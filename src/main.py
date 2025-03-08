import datetime
from queue_controller import QueueInterface
import time
from env import *
import traceback

if __name__ == "__main__":
    qi = QueueInterface(MONGO_URI, DB_NAME)
    qi.requeue_abandoned_trainigs()
    while True:
        try:
            qi.train_one()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"QueueInterface train_one() failed at {datetime.datetime.now()}")
            print(tb)
