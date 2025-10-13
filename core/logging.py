"""Centralised logging configuration helpers."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

__all__ = ["init_logging", "AnonymizingFormatter"]


class AnonymizingFormatter(logging.Formatter):
    """Formatter that masks potentially sensitive user paths."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._home = str(Path.home())

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting glue
        formatted = super().format(record)
        if self._home and self._home in formatted:
            formatted = formatted.replace(self._home, "~")
        return formatted


_FILE_HANDLER: Optional[RotatingFileHandler] = None
_CONSOLE_HANDLER: Optional[logging.Handler] = None
_FILE_PATH: Optional[Path] = None


_DEF_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEF_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _build_formatter() -> AnonymizingFormatter:
    return AnonymizingFormatter(fmt=_DEF_FORMAT, datefmt=_DEF_DATEFMT)


def init_logging(
    diagnostics_enabled: bool = False,
    *,
    level: int = logging.INFO,
    log_directory: Optional[Path | str] = None,
    log_filename: str = "application.log",
    max_bytes: int = 1_048_576,
    backup_count: int = 5,
) -> RotatingFileHandler:
    """Initialise root logging with rotating file and optional console output."""

    global _FILE_HANDLER, _CONSOLE_HANDLER, _FILE_PATH

    log_directory = Path(log_directory or Path("logs"))
    log_directory.mkdir(parents=True, exist_ok=True)
    target_path = log_directory / log_filename

    root_logger = logging.getLogger()
    formatter = _build_formatter()

    if _FILE_HANDLER is None or _FILE_PATH != target_path:
        if _FILE_HANDLER is not None:
            root_logger.removeHandler(_FILE_HANDLER)
            _FILE_HANDLER.close()
        file_handler = RotatingFileHandler(
            target_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        _FILE_HANDLER = file_handler
        _FILE_PATH = target_path

    if diagnostics_enabled:
        if _CONSOLE_HANDLER is None:
            _CONSOLE_HANDLER = logging.StreamHandler()
            _CONSOLE_HANDLER.setFormatter(_build_formatter())
            root_logger.addHandler(_CONSOLE_HANDLER)
        _CONSOLE_HANDLER.setLevel(logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
    else:
        if _CONSOLE_HANDLER is not None:
            root_logger.removeHandler(_CONSOLE_HANDLER)
            _CONSOLE_HANDLER.close()
            _CONSOLE_HANDLER = None
        root_logger.setLevel(level)

    return _FILE_HANDLER
