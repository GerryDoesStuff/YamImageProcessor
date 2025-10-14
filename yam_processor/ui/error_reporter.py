"""Helpers for presenting recoverable errors to the user."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from enum import Enum, auto
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

from yam_processor.data import sanitize_user_path

from .error_dialog import ErrorDialog


class ErrorResolution(Enum):
    """Outcome returned by :func:`present_error_report`."""

    CLOSED = auto()
    RETRY = auto()
    DISCARD_AUTOSAVE = auto()


def present_error_report(
    message: str,
    *,
    logger: logging.Logger,
    parent: Optional[QtWidgets.QWidget] = None,
    window_title: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
    component: Optional[str] = None,
    enable_retry: bool = False,
    retry_label: Optional[str] = None,
    enable_discard: bool = False,
    discard_label: Optional[str] = None,
    fallback_traceback: Optional[str] = None,
) -> ErrorResolution:
    """Present a detailed error dialog with optional recovery actions."""

    exc_info = sys.exc_info()
    raw_traceback = traceback.format_exc()
    if raw_traceback.strip() == "NoneType: None":
        raw_traceback = fallback_traceback or QtCore.QCoreApplication.translate(
            "ErrorReporter", "No traceback is available. Check the logs for more information."
        )

    sanitised_metadata = _sanitise_metadata(metadata)
    log_component = component or (sanitised_metadata.get("module") if sanitised_metadata else None) or logger.name
    log_extra: MutableMapping[str, Any] = {"component": log_component}
    if sanitised_metadata:
        log_extra["context"] = sanitised_metadata

    if exc_info[0] is None or raw_traceback.strip().startswith("No traceback"):
        logger.error(message, extra=log_extra)
    else:
        logger.exception(message, extra=log_extra)

    dialog = ErrorDialog(
        message,
        raw_traceback,
        parent=parent,
        metadata=sanitised_metadata or None,
        window_title=window_title,
    )

    resolution = ErrorResolution.CLOSED

    log_file = _discover_log_file()

    def _open_logs() -> None:
        if log_file is None:
            dialog.set_status_message(
                QtCore.QCoreApplication.translate("ErrorReporter", "Log directory is not available."),
                error=True,
            )
            return
        if not _open_log_directory(log_file):
            dialog.set_status_message(
                QtCore.QCoreApplication.translate("ErrorReporter", "Unable to open log directory: {path}").format(
                    path=_format_path(log_file.parent)
                ),
                error=True,
            )
            return
        dialog.set_status_message(
            QtCore.QCoreApplication.translate("ErrorReporter", "Opened log directory in your file browser."),
            error=False,
        )

    logs_button = dialog.add_action_button(
        QtCore.QCoreApplication.translate("ErrorReporter", "Open &Logs"), callback=_open_logs
    )
    if log_file is None:
        logs_button.setEnabled(False)
        logs_button.setToolTip(
            QtCore.QCoreApplication.translate("ErrorReporter", "Logging has not been initialised yet.")
        )

    if enable_retry:
        label = retry_label or QtCore.QCoreApplication.translate("ErrorReporter", "&Retry")

        def _trigger_retry() -> None:
            nonlocal resolution
            resolution = ErrorResolution.RETRY
            dialog.accept()

        dialog.add_action_button(label, callback=_trigger_retry)

    if enable_discard:
        label = discard_label or QtCore.QCoreApplication.translate("ErrorReporter", "&Discard autosave")

        def _trigger_discard() -> None:
            nonlocal resolution
            resolution = ErrorResolution.DISCARD_AUTOSAVE
            dialog.accept()

        dialog.add_action_button(
            label,
            role=QtWidgets.QDialogButtonBox.DestructiveRole,
            callback=_trigger_discard,
        )

    dialog.exec_()
    return resolution


def _discover_log_file() -> Optional[Path]:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                return Path(handler.baseFilename)
            except Exception:  # pragma: no cover - defensive guard
                continue
    return None


def _open_log_directory(log_file: Path) -> bool:
    directory = log_file.parent
    if not directory.exists():
        return False
    url = QtCore.QUrl.fromLocalFile(str(directory))
    return QtGui.QDesktopServices.openUrl(url)


def _sanitise_metadata(metadata: Optional[Mapping[str, object]]) -> dict[str, object]:
    if not metadata:
        return {}
    sanitised: dict[str, object] = {}
    for key, value in metadata.items():
        sanitised[str(key)] = _sanitise_value(value)
    return sanitised


def _sanitise_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _sanitise_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitise_value(item) for item in value]
    if isinstance(value, (os.PathLike, Path)):
        return _format_path(Path(value))
    if isinstance(value, str):
        maybe_path = _maybe_sanitise_path(value)
        return maybe_path or value
    return value


def _maybe_sanitise_path(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    has_separator = any(sep in candidate for sep in (os.sep, os.altsep) if sep)
    if not has_separator and not Path(candidate).is_absolute():
        return None
    return _format_path(Path(candidate))


def _format_path(path: Path) -> str:
    try:
        resolved = sanitize_user_path(path, must_exist=False, allow_directory=True, allow_file=True)
        return str(resolved)
    except Exception:  # pragma: no cover - sanitisation best effort
        path = path.expanduser()
        return path.name or str(path)


__all__ = ["ErrorResolution", "present_error_report"]
