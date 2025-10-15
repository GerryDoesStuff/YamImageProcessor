"""Threading helpers for background processing, progress reporting and cancellation.

The :class:`ThreadController` coordinates work submitted to a
``ThreadPoolExecutor`` while providing optional progress callbacks and
cooperative cancellation.  Callers can provide a ``progress_callback`` when
submitting work; the callback will receive floating point progress updates in
the ``[0.0, 1.0]`` range.  Tasks can also be supplied with either a
``cancel_token`` (callable or token-like object) or a ``cancellation_event`` to
request cooperative cancellation.  Internally a :class:`ThreadTask` wrapper is
used to monitor cancellation state while the user supplied worker executes and
will raise :class:`concurrent.futures.CancelledError` when cancellation is
requested.

Threaded workers can obtain a reference to the currently executing
``ThreadTask`` via :meth:`ThreadTask.current`.  The task exposes convenience
methods such as :meth:`ThreadTask.set_progress` and
:meth:`ThreadTask.raise_if_cancelled` that can be invoked inside long running
loops to emit fine grained progress notifications and respond to cancellation
requests.

For PyQt based UIs, :meth:`ThreadController.create_qt_signals` returns a small
adapter object exposing ``progress`` and ``cancel_requested`` signals.  The
adapter also offers ``progress_callback`` and ``cancellation_event`` accessors
that can be passed directly to :meth:`ThreadController.submit`, enabling UI
components to remain decoupled from the threading implementation while still
reacting to background task updates.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections import deque
from typing import Any, Callable, Deque, Optional

try:  # pragma: no cover - optional dependency helper
    from PyQt5.QtCore import QObject, pyqtSignal  # type: ignore
except ImportError:  # pragma: no cover - optional dependency helper
    try:
        from PySide6.QtCore import QObject, Signal as pyqtSignal  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency helper
        QObject = None  # type: ignore
        pyqtSignal = None  # type: ignore

_QT_AVAILABLE = QObject is not None and pyqtSignal is not None


Callback = Callable[[concurrent.futures.Future], None]
ProgressCallback = Callable[[float], None]


class ThreadTask:
    """Wrap a callable to provide progress callbacks and cooperative cancellation."""

    _task_local = threading.local()

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_token: Optional[Any] = None,
        cancellation_event: Optional[threading.Event] = None,
        poll_interval: float = 0.25,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._progress_callback = progress_callback
        self._cancel_token = cancel_token
        self._external_event = cancellation_event
        self._poll_interval = max(0.05, poll_interval)
        self._logger = logger or logging.getLogger(__name__)

        self._cancel_event = cancellation_event or threading.Event()
        self._completed_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._progress_value: float = 0.0
        self._cancelled = False
        self._future: Optional[concurrent.futures.Future] = None

    # ------------------------------------------------------------------
    # Cooperative API available to worker functions
    def set_progress(self, value: float) -> None:
        """Record new progress value and notify any callback immediately."""

        clamped = min(1.0, max(0.0, float(value)))
        self._progress_value = clamped
        if self._progress_callback is not None:
            try:
                self._progress_callback(clamped)
            except Exception:  # pragma: no cover - defensive callback guard
                self._logger.exception("Progress callback raised", extra={"component": "ThreadTask"})

    def raise_if_cancelled(self) -> None:
        """Raise :class:`CancelledError` if cancellation has been requested."""

        if self._is_cancelled(check_external=True):
            raise concurrent.futures.CancelledError()

    def is_cancelled(self) -> bool:
        """Return ``True`` when cancellation has been requested."""

        return self._is_cancelled(check_external=True)

    def cancellation_event(self) -> threading.Event:
        """Expose the underlying cancellation event for cooperative checks."""

        return self._cancel_event

    @classmethod
    def current(cls) -> Optional["ThreadTask"]:
        """Return the currently executing task for the calling thread, if any."""

        return getattr(cls._task_local, "task", None)

    # ------------------------------------------------------------------
    # Internal helpers
    def __call__(self) -> Any:
        type(self)._task_local.task = self
        self._start_monitor()
        try:
            self.set_progress(0.0)
            self.raise_if_cancelled()
            result = self._fn(*self._args, **self._kwargs)
            self.raise_if_cancelled()
            self.set_progress(1.0)
            return result
        finally:
            self._completed_event.set()
            self._stop_monitor()
            type(self)._task_local.task = None

    def bind_future(self, future: concurrent.futures.Future) -> None:
        """Associate the created future with this task for downstream access."""

        self._future = future

    def cancel(self) -> None:
        """Signal cancellation for the task."""

        self._cancel_event.set()

    def _start_monitor(self) -> None:
        should_monitor = (
            self._progress_callback is not None
            or self._cancel_token is not None
            or self._external_event is not None
        )
        if not should_monitor:
            return
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        if self._monitor_thread is None:
            return
        self._completed_event.set()
        self._monitor_thread.join()
        self._monitor_thread = None

    def _monitor_loop(self) -> None:
        while not self._completed_event.is_set():
            cancelled = self._is_cancelled(check_external=True)
            if cancelled:
                break
            if self._progress_callback is not None:
                try:
                    self._progress_callback(self._progress_value)
                except Exception:  # pragma: no cover - defensive callback guard
                    self._logger.exception("Progress callback raised", extra={"component": "ThreadTask"})
            self._completed_event.wait(self._poll_interval)

    def _is_cancelled(self, *, check_external: bool = False) -> bool:
        if self._cancelled:
            return True

        if check_external:
            if self._external_event is not None and self._external_event.is_set():
                self._cancel_event.set()
            if self._cancel_token is not None and self._evaluate_token(self._cancel_token):
                self._cancel_event.set()
            if self._future is not None and self._future.cancelled():
                self._cancel_event.set()

        if self._cancel_event.is_set():
            self._cancelled = True

        return self._cancelled

    def _evaluate_token(self, token: Any) -> bool:
        if callable(token):
            try:
                return bool(token())
            except TypeError:
                # Some cancellation tokens expose ``is_cancelled`` or similar attributes
                pass
        for attr in ("is_cancelled", "is_set", "cancelled", "done"):
            if hasattr(token, attr):
                value = getattr(token, attr)
                if callable(value):
                    return bool(value())
                return bool(value)
        return bool(token)


class ThreadController:
    """Coordinates threaded execution of background tasks."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending: Deque[concurrent.futures.Future] = deque()
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        self._pause_event = threading.Event()
        self._pause_event.set()

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        callback: Optional[Callback] = None,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_token: Optional[Any] = None,
        cancellation_event: Optional[threading.Event] = None,
        poll_interval: float = 0.25,
        **kwargs: Any,
    ) -> concurrent.futures.Future:
        """Submit a callable to execute in the background.

        Parameters mirror :class:`ThreadTask` making it possible to provide
        progress and cancellation hooks.
        """

        def _wait_for_resume() -> None:
            while not self._pause_event.wait(timeout=0.1):
                if cancellation_event is not None and cancellation_event.is_set():
                    raise concurrent.futures.CancelledError()
                if cancel_token is not None and self._evaluate_cancel_token(cancel_token):
                    raise concurrent.futures.CancelledError()

        def _run(*task_args: Any, **task_kwargs: Any) -> Any:
            _wait_for_resume()
            return fn(*task_args, **task_kwargs)

        task = ThreadTask(
            _run,
            args,
            kwargs,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            cancellation_event=cancellation_event,
            poll_interval=poll_interval,
            logger=self._logger,
        )
        future = self._executor.submit(task)
        task.bind_future(future)
        setattr(future, "task", task)
        with self._lock:
            self._pending.append(future)
        if callback is not None:
            future.add_done_callback(callback)
        future.add_done_callback(self._cleanup_future)
        self._logger.debug("Task submitted", extra={"component": "ThreadController", "pending": len(self._pending)})
        return future

    def pause(self) -> None:
        """Pause execution of background tasks."""

        self._pause_event.clear()
        self._logger.info("Background execution paused", extra={"component": "ThreadController"})

    def resume(self) -> None:
        """Resume execution of background tasks."""

        if not self._pause_event.is_set():
            self._pause_event.set()
            self._logger.info("Background execution resumed", extra={"component": "ThreadController"})

    def is_paused(self) -> bool:
        """Return ``True`` when background execution is paused."""

        return not self._pause_event.is_set()

    def cancel_all(self) -> None:
        """Attempt to cancel all pending tasks."""
        with self._lock:
            for future in list(self._pending):
                task = getattr(future, "task", None)
                if isinstance(task, ThreadTask):
                    task.cancel()
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

    @staticmethod
    def _evaluate_cancel_token(token: Any) -> bool:
        if callable(token):
            try:
                return bool(token())
            except TypeError:
                pass
        for attr in ("is_cancelled", "is_set", "cancelled", "done"):
            if hasattr(token, attr):
                value = getattr(token, attr)
                if callable(value):
                    return bool(value())
                return bool(value)
        return bool(token)

    # ------------------------------------------------------------------
    # Qt integration helpers
    @staticmethod
    def create_qt_signals(parent: Optional[Any] = None) -> "QtThreadSignals":
        """Create a PyQt/PySide friendly signal adapter for thread tasks.

        Returns an object exposing ``progress`` and ``cancel_requested`` signals
        alongside helper methods ``progress_callback`` and
        ``cancellation_event`` that can be supplied to :meth:`submit`.  A
        :class:`RuntimeError` is raised if no Qt binding is available.
        """

        if not _QT_AVAILABLE:  # pragma: no cover - requires Qt
            raise RuntimeError("Qt bindings are not available in this environment")
        return QtThreadSignals(parent)


if _QT_AVAILABLE:  # pragma: no cover - requires Qt bindings

    class QtThreadSignals(QObject):
        """Minimal Qt signal adapter for thread progress and cancellation."""

        progress = pyqtSignal(float)
        cancel_requested = pyqtSignal()

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self._cancel_event = threading.Event()

        # Signals -------------------------------------------------------------
        def progress_callback(self) -> ProgressCallback:
            """Return a callback that emits the ``progress`` signal."""

            def emit_progress(value: float) -> None:
                self.progress.emit(float(value))

            return emit_progress

        def cancellation_event(self) -> threading.Event:
            """Return an event that is set when ``request_cancel`` is invoked."""

            return self._cancel_event

        def request_cancel(self) -> None:
            """Notify listeners that cancellation has been requested."""

            self._cancel_event.set()
            self.cancel_requested.emit()

else:  # pragma: no cover - requires Qt bindings

    class QtThreadSignals:  # type: ignore[override]
        """Placeholder used when Qt bindings are unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Qt bindings are not available in this environment")

