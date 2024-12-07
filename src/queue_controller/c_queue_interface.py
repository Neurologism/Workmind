from pymongo import MongoClient
import time
from datetime import datetime, timedelta, timezone
from tensorflow_middleware import WhitemindProject
import threading


class QueueInterface:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.db_training_queue = self.db["queueitems"]
        self.db_models = self.db["tasks"]
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

        self.model = self.db_models.find_one({"_id": queue_item["taskId"]})
        if self.model is None:
            raise ValueError("Model does not exist.")

        self.db_models.update_one(
            {"_id": self.model["_id"]},
            {
                "$set": {
                    "status": "training",
                    "datelastUpdated": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )[:-3]
                    + "Z",
                    "dateStarted": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )[:-3]
                    + "Z",
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
            print(f"Error during training, check database for details: {e}")
            self.db_models.update_one(
                {"_id": self.model["_id"]},
                {
                    "$set": {
                        "status": "error",
                        "datelastUpdated": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.%f"
                        )[:-3]
                        + "Z",
                        "error": str(e),
                    }
                },
            )
            print(f"Error: ")
        finally:
            self.db_updater_running = False

        db_updater_thread.join()

        self.db_models.update_one(
            {"_id": self.model["_id"]},
            {
                "$set": {
                    "status": "finished",
                    "datelastUpdated": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )[:-3]
                    + "Z",
                    "dateFinished": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )[:-3]
                    + "Z",
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
                        "datelastUpdated": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.%f"
                        )[:-3]
                        + "Z",
                    },
                    "$push": {"output": {"$each": payloads}},
                },
            )
            time.sleep(1)

    def requeue_abandoned_trainigs(self) -> None:
        print("Searching for abandoned trainings...")
        found = False
        for model in self.db_models.find({"status": "training"}):
            lastUpdated = datetime.strptime(
                model["datelastUpdated"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
            if lastUpdated < datetime.now(timezone.utc) - timedelta(minutes=1):
                self.db_models.update_one(
                    {"_id": model["_id"]},
                    {
                        "$set": {
                            "status": "queued",
                            "datelastUpdated": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%S.%f"
                            )[:-3]
                            + "Z",
                        }
                    },
                )

                self.db_training_queue.insert_one(
                    {
                        "taskId": model["_id"],
                    }
                )

                found = True

        if found:
            print("Found and requeued abandoned trainings.")
        else:
            print("No abandoned trainings found.")
