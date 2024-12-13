import dotenv
import os
from queue_controller import QueueInterface
import time

dotenv.load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise EnvironmentError("MONGO_URI is not set")
DB_NAME = os.getenv("DB_NAME") or "backmind"

if __name__ == "__main__":
    qi = QueueInterface(MONGO_URI, DB_NAME)
    qi.requeue_abandoned_trainigs()
    while True:
        try:
            qi.train_one()
        except Exception as e:
            print(f"Error during training: {e}")
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
