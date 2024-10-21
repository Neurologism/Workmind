import pymongo
from c_queue_item import QueueItem


class QueueInterface:
    def __init__(self) -> None:
        # connect to mongodb
        pass

    def get_queue_item(self) -> QueueItem:
        pass
