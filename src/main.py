import dotenv
import os
import datetime
from queue_controller import QueueInterface
import time
from env import *

if __name__ == "__main__":
    qi = QueueInterface(MONGO_URI, DB_NAME)
    qi.requeue_abandoned_trainigs()
    while True:
        try:
            qi.train_one()
        except Exception as e:
            print(f"At {datetime.datetime.now()} an Error occurred: {e}")
            print("Reloading queue interface...")
            while True:
                try:
                    qi = QueueInterface(MONGO_URI, DB_NAME)
                    qi.requeue_abandoned_trainigs()
                    break
                except Exception as e:
                    print(f"Error during reloading: {e}")
                    print("Sleeping 10 seconds...")
                    time.sleep(10)
