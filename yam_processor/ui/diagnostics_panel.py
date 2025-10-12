"""Qt diagnostics panel combining log output and task monitoring."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PyQt5 import QtCore, QtWidgets  # type: ignore

from yam_processor.core.threading import ThreadController


class QtLogHandler(QtCore.QObject, logging.Handler):
    """Logging handler that emits formatted messages to Qt slots."""

    messageEmitted = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        QtCore.QObject.__init__(self, parent)
        logging.Handler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - requires Qt event loop
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive against formatting errors
            message = record.getMessage()
        self.messageEmitted.emit(message)


class DiagnosticsPanel(QtWidgets.QWidget):
    """Widget displaying log output alongside background task information."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._log_handler = QtLogHandler(self)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.messageEmitted.connect(self._append_log_message)
        self._attached_logger: Optional[logging.Logger] = None
        self._thread_controller: Optional[ThreadController] = None
        self._tasks: Dict[str, QtWidgets.QTreeWidgetItem] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def log_handler(self) -> QtLogHandler:
        """Return the Qt aware log handler used by the panel."""

        return self._log_handler

    def attach_to_logger(self, logger: Optional[logging.Logger]) -> None:
        """Attach the panel handler to ``logger`` replacing any previous binding."""

        if self._attached_logger is logger:
            return
        if self._attached_logger is not None:
            self._attached_logger.removeHandler(self._log_handler)
        self._attached_logger = logger
        if logger is not None:
            logger.addHandler(self._log_handler)

    def detach_from_logger(self) -> None:
        """Remove the panel handler from the currently attached logger."""

        if self._attached_logger is not None:
            self._attached_logger.removeHandler(self._log_handler)
        self._attached_logger = None

    def clear_logs(self) -> None:
        """Remove all log entries from the display."""

        self.log_view.clear()

    def focus_logs(self) -> None:
        """Focus the log view while ensuring the last entry is visible."""

        self.log_view.setFocus(QtCore.Qt.OtherFocusReason)
        self._scroll_to_bottom()

    # ------------------------------------------------------------------
    # Task lifecycle API
    # ------------------------------------------------------------------
    @QtCore.pyqtSlot(str, str)
    def register_task(self, task_id: str, description: str) -> None:
        """Register a new task with ``task_id`` and human readable ``description``."""

        item = self._ensure_task_item(task_id, description)
        item.setText(1, self.tr("Queued"))
        item.setText(2, self.tr("0%"))

    @QtCore.pyqtSlot(str, str)
    def update_task_status(self, task_id: str, status: str) -> None:
        """Update ``task_id`` with a new textual ``status``."""

        item = self._ensure_task_item(task_id)
        item.setText(1, status)

    @QtCore.pyqtSlot(str, float)
    def update_task_progress(self, task_id: str, progress: float) -> None:
        """Update ``task_id`` progress as a percentage ``progress`` in the range [0, 1]."""

        item = self._ensure_task_item(task_id)
        clamped = max(0.0, min(progress, 1.0))
        item.setText(2, f"{clamped * 100:.0f}%")

    @QtCore.pyqtSlot(str)
    def complete_task(self, task_id: str) -> None:
        """Mark ``task_id`` as completed."""

        item = self._tasks.get(task_id)
        if item is not None:
            item.setText(1, self.tr("Completed"))
            item.setText(2, self.tr("100%"))

    @QtCore.pyqtSlot(str)
    def remove_task(self, task_id: str) -> None:
        """Remove ``task_id`` from the task table."""

        item = self._tasks.pop(task_id, None)
        if item is not None:
            index = self.task_view.indexOfTopLevelItem(item)
            if index != -1:
                self.task_view.takeTopLevelItem(index)

    def set_thread_controller(self, controller: Optional[ThreadController]) -> None:
        """Store a reference to the shared :class:`ThreadController`."""

        self._thread_controller = controller

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        log_group = QtWidgets.QGroupBox(self.tr("Log Output"), self)
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_view = QtWidgets.QPlainTextEdit(log_group)
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_view.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, 3)

        task_group = QtWidgets.QGroupBox(self.tr("Background Tasks"), self)
        task_layout = QtWidgets.QVBoxLayout(task_group)
        self.task_view = QtWidgets.QTreeWidget(task_group)
        self.task_view.setColumnCount(3)
        self.task_view.setHeaderLabels(
            [self.tr("Task"), self.tr("Status"), self.tr("Progress")]
        )
        self.task_view.setRootIsDecorated(False)
        self.task_view.setUniformRowHeights(True)
        self.task_view.setAlternatingRowColors(True)
        task_layout.addWidget(self.task_view)
        layout.addWidget(task_group, 2)

    def _append_log_message(self, message: str) -> None:
        at_bottom = self._is_scrolled_to_bottom()
        self.log_view.appendPlainText(message)
        if at_bottom:
            self._scroll_to_bottom()

    def _ensure_task_item(
        self, task_id: str, description: Optional[str] = None
    ) -> QtWidgets.QTreeWidgetItem:
        item = self._tasks.get(task_id)
        if item is None:
            item = QtWidgets.QTreeWidgetItem(
                [description or task_id, self.tr("Queued"), self.tr("0%")]
            )
            item.setData(0, QtCore.Qt.UserRole, task_id)
            self.task_view.addTopLevelItem(item)
            self._tasks[task_id] = item
        elif description is not None:
            item.setText(0, description)
        return item

    def _is_scrolled_to_bottom(self) -> bool:
        scroll_bar = self.log_view.verticalScrollBar()
        return scroll_bar.value() >= scroll_bar.maximum()

    def _scroll_to_bottom(self) -> None:
        scroll_bar = self.log_view.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
