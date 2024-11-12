from pymongo import MongoClient
import time
from datetime import datetime
from tensorflow_middleware import WhitemindProject


class QueueInterface:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client(db_name)
        self.db_training_queue = self.db["training_queue"]
        self.db_models = self.db["models"]

    def train_one(self) -> None:
        queue_item = self.db_training_queue.find_one_and_delete({})

        if queue_item is None:
            print("Queue is empty, sleeping...")
            while queue_item is None:
                time.sleep(5)
                queue_item = self.db_training_queue.find_one_and_delete({})

        print("Queue item found.")

        model = self.db_models.find_one({"_id": queue_item.model_id})
        if model is None:
            raise ValueError("Model does not exist.")

        self.db_models.update_one(
            {"_id": queue_item.model_id},
            {
                "$set": {
                    "status": "training",
                    "last_updated": datetime.now().timestamp(),
                    "started_at": datetime.now().timestamp(),
                }
            },
        )

        try:
            project = WhitemindProject(model.task)
            project.execute()
        except Exception as e:
            print("Error during trainig")
            raise e
