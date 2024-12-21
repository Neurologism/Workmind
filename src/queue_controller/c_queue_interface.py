from pymongo import MongoClient
import time
from datetime import datetime, timedelta, timezone
from tensorflow_middleware import WhitemindProject
import multiprocessing


def run_whitemind_project(conn, task):
    try:
        project = WhitemindProject(task, lambda payload: conn.send(payload))
        project.execute()
    except Exception as e:
        print(f"Error during training, check database for details: {e}")
        conn.send(
            {
                "type": "error",
                "message": str(e),
            }
        )
        time.sleep(2)
    finally:
        conn.close()


def db_updater(conn, mongo_uri, db_name, id):
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[db_name]
    db_models = db["tasks"]

    while 1:
        try:
            payloads = []
            while conn.poll():
                payloads.append(conn.recv())
                if "type" in payloads[-1] and payloads[-1]["type"] == "error":
                    db_models.update_one(
                        {"_id": id},
                        {
                            "$set": {
                                "status": "error",
                                "datelastUpdated": datetime.now(timezone.utc).strftime(
                                    "%Y-%m-%dT%H:%M:%S.%f"
                                )[:-3]
                                + "Z",
                                "error": payloads[-1]["message"],
                            }
                        },
                    )
                    return
            db_models.update_one(
                {"_id": id},
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
        except Exception as e:
            print(f"Error during updating database: {e}")
            time.sleep(2)


class QueueInterface:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.db_training_queue = self.db["queueitems"]
        self.db_models = self.db["tasks"]
        self.db_users = self.db["users"]
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

        parent_conn, child_conn = multiprocessing.Pipe()

        p1 = multiprocessing.Process(
            target=run_whitemind_project, args=(parent_conn, self.model["task"])
        )
        p2 = multiprocessing.Process(
            target=db_updater,
            args=(child_conn, self.mongo_uri, self.db_name, self.model["_id"]),
        )

        p1.start()
        p2.start()

        while p1.is_alive() and p2.is_alive():
            time.sleep(1)
            model = self.db_models.find_one({"_id": self.model["_id"]})
            if model["status"] == "stopped":
                print("\nTraining stopped.")
                break

            user = self.db_users.find_one({"_id": model["ownerId"]})
            credits = user["remainingCredits"] if "remainingCredits" in user else 0
            if credits <= 0:
                print("\nInsufficient credits.")
                self.db_models.update_one(
                    {"_id": self.model["_id"]},
                    {
                        "$set": {
                            "status": "stopped",
                            "datelastUpdated": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%S.%f"
                            )[:-3]
                            + "Z",
                            "error": "Insufficient credits",
                        }
                    },
                )
            else:
                self.db_users.update_one(
                    {"_id": model["ownerId"]},
                    {"$set": {"remainingCredits": credits - 1}},
                )

        if p1.is_alive():
            p1.terminate()

        if p2.is_alive():
            p2.terminate()

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
