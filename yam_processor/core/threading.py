"""Threading infrastructure for background processing and task management."""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections import deque
from typing import Any, Callable, Deque, Optional


Callback = Callable[[concurrent.futures.Future], None]


class ThreadController:
    """Coordinates threaded execution of background tasks."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending: Deque[concurrent.futures.Future] = deque()
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def submit(self, fn: Callable[..., Any], *args: Any, callback: Optional[Callback] = None, **kwargs: Any) -> concurrent.futures.Future:
        """Submit a callable to execute in the background."""
        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._pending.append(future)
        if callback is not None:
            future.add_done_callback(callback)
        future.add_done_callback(self._cleanup_future)
        self._logger.debug("Task submitted", extra={"component": "ThreadController", "pending": len(self._pending)})
        return future

    def cancel_all(self) -> None:
        """Attempt to cancel all pending tasks."""
        with self._lock:
            for future in list(self._pending):
                future.cancel()
            self._pending.clear()
        self._logger.info("All background tasks cancelled", extra={"component": "ThreadController"})

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the executor and cancel pending tasks."""
        self.cancel_all()
        self._executor.shutdown(wait=wait)
        self._logger.info("Thread controller shutdown", extra={"component": "ThreadController"})

    def _cleanup_future(self, future: concurrent.futures.Future) -> None:
        with self._lock:
            if future in self._pending:
                self._pending.remove(future)
        self._logger.debug("Task finished", extra={"component": "ThreadController", "pending": len(self._pending)})
