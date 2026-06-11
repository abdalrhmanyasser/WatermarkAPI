import threading

class ProgressTracker:
    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()

    def update_progress(self, task_id: str, percentage: int, status: str):
        with self._lock:
            self._tasks[task_id] = {
                "progress": percentage, 
                "status": status, 
                "result": self._tasks.get(task_id, {}).get("result")
            }

    def complete_task(self, task_id: str, result: dict):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["progress"] = 100
                self._tasks[task_id]["status"] = "Completed"
                self._tasks[task_id]["result"] = result

    def get_status(self, task_id: str):
        with self._lock:
            return self._tasks.get(task_id, {"progress": 0, "status": "Not Found", "result": None})

# Global instance to be imported across the app
progress_tracker = ProgressTracker()