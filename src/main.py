import dotenv
import os
from queue_controller import QueueInterface

dotenv.load_dotenv()

env = {}
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise EnvironmentError("MONGO_URI is not set")
DB_NAME = os.getenv("DB_NAME") or "backmind"

if __name__ == "__main__":
    qi = QueueInterface(MONGO_URI, DB_NAME)
    while True:
        qi.train_one()
