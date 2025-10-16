"""Qt-aware thread controller for executing pipeline work asynchronously."""

from __future__ import annotations

import inspect
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PyQt5 import QtCore


class OperationCancelled(RuntimeError):
    """Raised internally when a task is cancelled before completion."""


class _WorkerSignals(QtCore.QObject):
    """Signals emitted from worker threads and forwarded to the UI."""

    progress = QtCore.pyqtSignal(int)
    intermediate = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)
    canceled = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(Exception, str)


class _FunctionRunnable(QtCore.QRunnable):
    """Wrap a callable for execution in a :class:`QThreadPool`."""

    def __init__(
        self,
        function: Callable[[threading.Event, Callable[[int], None]], Any],
        cancel_event: threading.Event,
        signals: _WorkerSignals,
        pause_event: threading.Event,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._function = function
        self._cancel_event = cancel_event
        self._signals = signals
        self._pause_event = pause_event
        self._accepts_intermediate = self._detect_intermediate_support(function)

    @QtCore.pyqtSlot()
    def run(self) -> None:  # pragma: no cover - executed on worker thread
        try:
            while not self._pause_event.wait(0.1):
                if self._cancel_event.is_set():
                    raise OperationCancelled()
            if self._accepts_intermediate:
                result = self._function(
                    self._cancel_event,
                    self._signals.progress.emit,
                    self._signals.intermediate.emit,
                )
            else:
                result = self._function(
                    self._cancel_event,
                    self._signals.progress.emit,
                )
        except OperationCancelled:
            self._signals.canceled.emit()
        except Exception as exc:  # pragma: no cover - defensive handling
            traceback_str = traceback.format_exc()
            self._signals.failed.emit(exc, traceback_str)
        else:
            self._signals.finished.emit(result)

    @staticmethod
    def _detect_intermediate_support(
        function: Callable[[threading.Event, Callable[[int], None]], Any]
    ) -> bool:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            return False
        parameters = list(signature.parameters.values())
        if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
            return True
        positional = [
            param
            for param in parameters
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 3


@dataclass
class _TaskCallbacks:
    finished: Optional[Callable[[Any], None]] = None
    canceled: Optional[Callable[[], None]] = None
    failed: Optional[Callable[[Exception, str], None]] = None


class ThreadController(QtCore.QObject):
    """Coordinate background work and forward results to the Qt event loop."""

    task_started = QtCore.pyqtSignal(str)
    task_progress = QtCore.pyqtSignal(int)
    task_intermediate = QtCore.pyqtSignal(object)
    task_finished = QtCore.pyqtSignal(object)
    task_canceled = QtCore.pyqtSignal()
    task_failed = QtCore.pyqtSignal(Exception, str)

    def __init__(
        self,
        *,
        parent: Optional[QtCore.QObject] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        if max_workers is not None:
            self._thread_pool.setMaxThreadCount(max_workers)
        self._cancel_event: Optional[threading.Event] = None
        self._callbacks: _TaskCallbacks = _TaskCallbacks()
        self._signals: Optional[_WorkerSignals] = None
        self._task_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._paused = False

    # ------------------------------------------------------------------
    # Public API
    def is_running(self) -> bool:
        """Return ``True`` when a background task is active."""

        return self._cancel_event is not None

    def is_paused(self) -> bool:
        """Return ``True`` if new tasks should wait before executing."""

        with self._task_lock:
            return self._paused

    def pause(self) -> None:
        """Prevent tasks from executing until :meth:`resume` is called."""

        with self._task_lock:
            if self._paused:
                return
            self._paused = True
            self._pause_event.clear()

    def resume(self) -> None:
        """Allow paused tasks to resume execution."""

        with self._task_lock:
            if not self._paused:
                return
            self._paused = False
            self._pause_event.set()

    def cancel(self) -> None:
        """Request cancellation of the currently running task, if any."""

        with self._task_lock:
            if self._cancel_event is not None:
                self._cancel_event.set()

    def shutdown(self) -> None:
        """Drain the underlying thread pool."""

        self.resume()
        self.cancel()
        self._thread_pool.waitForDone()

    def run_task(
        self,
        function: Callable[[threading.Event, Callable[[int], None]], Any],
        *,
        description: str = "",
        on_finished: Optional[Callable[[Any], None]] = None,
        on_canceled: Optional[Callable[[], None]] = None,
        on_failed: Optional[Callable[[Exception, str], None]] = None,
        on_intermediate: Optional[Callable[[Any], None]] = None,
    ) -> _WorkerSignals:
        """Execute ``function`` on a worker thread."""

        cancel_event = threading.Event()
        signals = _WorkerSignals()
        runnable = _FunctionRunnable(function, cancel_event, signals, self._pause_event)

        with self._task_lock:
            self._cancel_event = cancel_event
            self._signals = signals
            self._callbacks = _TaskCallbacks(
                finished=on_finished, canceled=on_canceled, failed=on_failed
        )

        signals.progress.connect(self.task_progress.emit)
        signals.intermediate.connect(self.task_intermediate.emit)
        signals.finished.connect(self._on_task_finished)
        signals.canceled.connect(self._on_task_canceled)
        signals.failed.connect(self._on_task_failed)
        if on_intermediate is not None:
            signals.intermediate.connect(on_intermediate)

        self.task_started.emit(description)
        self._thread_pool.start(runnable)
        return signals

    def run_pipeline(
        self,
        pipeline: Any,
        image: Any,
        *,
        description: str = "",
        on_finished: Optional[Callable[[Any], None]] = None,
        on_canceled: Optional[Callable[[], None]] = None,
        on_failed: Optional[Callable[[Exception, str], None]] = None,
        on_intermediate: Optional[Callable[[Any], None]] = None,
    ) -> _WorkerSignals:
        """Execute ``pipeline.apply`` using a worker thread."""

        steps = list(getattr(pipeline, "iter_enabled_steps", lambda: [])())
        total_steps = len(steps) or 1

        def _execute(
            cancel_event: threading.Event,
            progress: Callable[[int], None],
            intermediate: Optional[Callable[[Any], None]] = None,
        ) -> Any:
            result = image.copy() if hasattr(image, "copy") else image
            if not steps:
                if cancel_event.is_set():
                    raise OperationCancelled()
                progress(100)
                return pipeline.apply(result)

            for index, step in enumerate(steps, start=1):
                if cancel_event.is_set():
                    raise OperationCancelled()
                result = step.apply(result)
                progress(int(index * 100 / total_steps))
            return result

        return self.run_task(
            _execute,
            description=description,
            on_finished=on_finished,
            on_canceled=on_canceled,
            on_failed=on_failed,
            on_intermediate=on_intermediate,
        )

    # ------------------------------------------------------------------
    # Internal slots
    @QtCore.pyqtSlot(object)
    def _on_task_finished(self, result: Any) -> None:
        callbacks = self._finalise_task()
        self.task_finished.emit(result)
        if callbacks.finished is not None:
            callbacks.finished(result)

    @QtCore.pyqtSlot()
    def _on_task_canceled(self) -> None:
        callbacks = self._finalise_task()
        self.task_canceled.emit()
        if callbacks.canceled is not None:
            callbacks.canceled()

    @QtCore.pyqtSlot(Exception, str)
    def _on_task_failed(self, error: Exception, stack: str) -> None:
        callbacks = self._finalise_task()
        self.task_failed.emit(error, stack)
        if callbacks.failed is not None:
            callbacks.failed(error, stack)

    def _finalise_task(self) -> _TaskCallbacks:
        with self._task_lock:
            callbacks = self._callbacks
            self._callbacks = _TaskCallbacks()
            self._cancel_event = None
            self._signals = None
        return callbacks


__all__ = ["OperationCancelled", "ThreadController"]

