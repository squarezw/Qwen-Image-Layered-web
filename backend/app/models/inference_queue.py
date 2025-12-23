"""
Inference Queue - Manages GPU inference tasks sequentially
"""
import asyncio
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid


@dataclass
class TaskInfo:
    """Information about a queued task"""
    task_id: str
    task_type: str
    task_func: Callable
    kwargs: dict
    status: str = "queued"  # queued, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    message: str = ""


class InferenceQueue:
    """
    Manages GPU inference queue with sequential processing
    Ensures only one GPU task runs at a time for optimal memory usage
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.results: Dict[str, Any] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        self.worker_started = False
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None

    async def submit_task(
        self,
        task_id: str,
        task_type: str,
        task_func: Callable,
        **kwargs
    ) -> str:
        """
        Submit a task to the inference queue

        Args:
            task_id: Unique task identifier
            task_type: Type of task (decomposition/editing)
            task_func: Function to execute
            **kwargs: Arguments to pass to task_func

        Returns:
            task_id: The task identifier
        """
        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            task_func=task_func,
            kwargs=kwargs,
            status="queued",
            created_at=datetime.utcnow()
        )

        async with self._lock:
            self.active_tasks[task_id] = task_info
            await self.queue.put(task_info)

        # Start worker if not already running
        if not self.worker_started:
            self._worker_task = asyncio.create_task(self._worker())
            self.worker_started = True

        return task_id

    async def _worker(self):
        """
        Background worker that processes queue sequentially
        Only one task is processed at a time to manage GPU memory
        """
        while True:
            try:
                # Get next task from queue
                task_info = await self.queue.get()
                task_id = task_info.task_id

                try:
                    # Update status to processing
                    task_info.status = "processing"
                    task_info.started_at = datetime.utcnow()
                    await self._notify_progress(task_id, 0, "Starting inference...")

                    # Create progress callback for this task
                    def progress_callback(progress: float, message: str):
                        task_info.progress = progress
                        task_info.message = message
                        # Schedule notification in event loop
                        asyncio.create_task(self._notify_progress(task_id, progress, message))

                    # Execute task (in thread pool to avoid blocking)
                    result = await asyncio.to_thread(
                        self._execute_with_progress,
                        task_info,
                        progress_callback
                    )

                    # Store result
                    self.results[task_id] = result
                    task_info.status = "completed"
                    task_info.completed_at = datetime.utcnow()
                    await self._notify_progress(task_id, 100, "Completed")

                except Exception as e:
                    # Handle errors
                    task_info.status = "failed"
                    task_info.error = str(e)
                    task_info.completed_at = datetime.utcnow()
                    await self._notify_progress(task_id, -1, f"Error: {e}")
                    print(f"Task {task_id} failed: {e}")

                finally:
                    # Mark task as done in queue
                    self.queue.task_done()

            except Exception as e:
                print(f"Worker error: {e}")
                # Continue processing other tasks

    def _execute_with_progress(self, task_info: TaskInfo, progress_callback: Callable):
        """
        Execute task with progress tracking

        Args:
            task_info: Task information
            progress_callback: Callback for progress updates
        """
        # Add progress callback to kwargs
        kwargs = task_info.kwargs.copy()
        kwargs['progress_callback'] = progress_callback

        # Execute task function
        result = task_info.task_func(**kwargs)
        return result

    async def _notify_progress(self, task_id: str, progress: float, message: str):
        """
        Notify all registered progress callbacks for a task

        Args:
            task_id: Task identifier
            progress: Progress percentage (0-100)
            message: Progress message
        """
        if task_id in self.progress_callbacks:
            for callback in self.progress_callbacks[task_id]:
                try:
                    await callback(progress, message)
                except Exception as e:
                    print(f"Progress callback error: {e}")

    def register_progress_callback(self, task_id: str, callback: Callable):
        """
        Register a progress callback for a task

        Args:
            task_id: Task identifier
            callback: Async function to call with (progress, message)
        """
        if task_id not in self.progress_callbacks:
            self.progress_callbacks[task_id] = []
        self.progress_callbacks[task_id].append(callback)

    def unregister_progress_callback(self, task_id: str, callback: Callable):
        """
        Unregister a progress callback for a task

        Args:
            task_id: Task identifier
            callback: The callback to remove
        """
        if task_id in self.progress_callbacks:
            try:
                self.progress_callbacks[task_id].remove(callback)
                if not self.progress_callbacks[task_id]:
                    del self.progress_callbacks[task_id]
            except ValueError:
                pass

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a task

        Args:
            task_id: Task identifier

        Returns:
            Dictionary with task status information or None if not found
        """
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            return None

        return {
            "task_id": task_id,
            "status": task_info.status,
            "progress": task_info.progress,
            "message": task_info.message,
            "created_at": task_info.created_at.isoformat(),
            "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
            "completed_at": task_info.completed_at.isoformat() if task_info.completed_at else None,
            "error": task_info.error
        }

    def get_result(self, task_id: str) -> Optional[Any]:
        """
        Get result of a completed task

        Args:
            task_id: Task identifier

        Returns:
            Task result or None if not found/completed
        """
        return self.results.get(task_id)

    def get_queue_position(self, task_id: str) -> int:
        """
        Get position of a task in the queue

        Args:
            task_id: Task identifier

        Returns:
            Position in queue (0 = currently processing, -1 = not found)
        """
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            return -1

        if task_info.status == "processing":
            return 0

        # Count tasks ahead in queue
        position = 1
        for other_task in list(self.active_tasks.values()):
            if other_task.status == "queued" and other_task.created_at < task_info.created_at:
                position += 1

        return position

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    async def cleanup_old_results(self, max_age_hours: int = 24):
        """
        Clean up old results to free memory

        Args:
            max_age_hours: Maximum age of results to keep
        """
        now = datetime.utcnow()
        to_remove = []

        for task_id, task_info in self.active_tasks.items():
            if task_info.completed_at:
                age = (now - task_info.completed_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(task_id)

        for task_id in to_remove:
            self.active_tasks.pop(task_id, None)
            self.results.pop(task_id, None)
            self.progress_callbacks.pop(task_id, None)
