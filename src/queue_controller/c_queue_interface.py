from pymongo import MongoClient
from c_queue_item import QueueItem


class QueueInterface:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client(db_name)
        self.db_training_queue = self.db["training_queue"]
        self.db_models = self.db["models"]

    def get_queue_item(self) -> QueueItem | None:
        queue_item = self.db_training_queue.find_one_and_delete({})
        if queue_item is None:
            return queue_item
        return QueueItem(queue_item)
