class QueueItem:
    def __init__(self, document: dict) -> None:
        self.status = document.status
        self.output = document.output
        self.task = document.task
        self.last_updated_at = document.last_updated_at
        self.queued_at = document.queued_at
        self.started_at = document.started_at
        self.finished_at = document.finished_at
        self.error = document.error
        self.project_id = document.project_id
