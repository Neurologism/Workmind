from pymongo import MongoClient
import time
from datetime import datetime
from tensorflow_middleware import WhitemindProject
import threading


class QueueInterface:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.db_training_queue = self.db["training_queue"]
        self.db_models = self.db["models"]
        self.db_updater_running = False
        self.logging_payloads = []
        self.model = None

    def train_one(self) -> None:
        queue_item = self.db_training_queue.find_one_and_delete({})

        if queue_item is None:
            print("Queue is empty, sleeping...")
            while queue_item is None:
                time.sleep(5)
                queue_item = self.db_training_queue.find_one_and_delete({})
        print("Queue item found.")

        self.model = self.db_models.find_one({"_id": queue_item["model_id"]})
        if self.model is None:
            raise ValueError("Model does not exist.")

        self.db_models.update_one(
            {"_id": queue_item["model_id"]},
            {
                "$set": {
                    "status": "training",
                    "last_updated": datetime.now().timestamp(),
                    "started_at": datetime.now().timestamp(),
                }
            },
        )

        project = WhitemindProject(self.model["task"], self.define_log())
        self.db_updater_running = True
        try:
            db_updater_thread = threading.Thread(target=self.db_updater)
            db_updater_thread.start()
            project.execute()
        except Exception as e:
            print("Error during trainig, check database for details")
            self.db_models.update_one(
                {"_id": queue_item["model_id"]},
                {
                    "$set": {
                        "status": "error",
                        "last_updated": datetime.now().timestamp(),
                        "error": str(e),
                    }
                },
            )
        finally:
            self.db_updater_running = False

        db_updater_thread.join()

        self.db_models.update_one(
            {"_id": queue_item["model_id"]},
            {
                "$set": {
                    "status": "finished",
                    "last_updated": datetime.now().timestamp(),
                    "finished_at": datetime.now().timestamp(),
                }
            },
        )

    def define_log(self) -> None:
        def log(payload):
            self.logging_payloads.append(payload)

        return log

    def db_updater(self) -> None:
        while self.db_updater_running or len(self.logging_payloads):
            payloads = self.logging_payloads
            self.logging_payloads = []
            self.db_models.update_one(
                {"_id": self.model["_id"]},
                {
                    "$set": {
                        "last_updated": datetime.now().timestamp(),
                    },
                    "$push": {"output": {"$each": payloads}},
                },
            )
            time.sleep(1)
