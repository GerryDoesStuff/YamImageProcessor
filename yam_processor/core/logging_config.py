"""Centralized logging configuration for YamImageProcessor."""
from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FILENAME = "yam_image_processor.log"
DEFAULT_LOG_DIRNAME = "logs"


class _ComponentSafeFormatter(logging.Formatter):
    """Ensures log records always include a component field for formatting."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting is hard to unit test reliably
        if not hasattr(record, "component"):
            record.component = record.name
        return super().format(record)


@dataclass
class LoggingOptions:
    """Runtime options for configuring the logging subsystem."""

    log_directory: Optional[os.PathLike] = None
    level: int = logging.INFO
    enable_console: bool = True
    developer_diagnostics: bool = False
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


class LoggingConfigurator:
    """Configures application-wide logging with rotation and anonymised output."""

    def __init__(self, options: Optional[LoggingOptions] = None) -> None:
        self.options = options or LoggingOptions()
        self.logger = logging.getLogger()

    def configure(self) -> None:
        """Initialise logging handlers based on provided options."""
        level = logging.DEBUG if self.options.developer_diagnostics else self.options.level
        self.logger.setLevel(level)
        self._clear_existing_handlers()

        log_path = self._determine_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=self.options.max_bytes,
            backupCount=self.options.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(self._build_formatter(anonymise=True))
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)

        if self.options.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._build_formatter(anonymise=not self.options.developer_diagnostics))
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)

        self.logger.debug("Logging configured", extra={"component": "LoggingConfigurator"})

    def _determine_log_path(self) -> Path:
        base_dir = (
            Path(self.options.log_directory)
            if self.options.log_directory is not None
            else Path.home() / DEFAULT_LOG_DIRNAME
        )
        return base_dir / DEFAULT_LOG_FILENAME

    def _clear_existing_handlers(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    @staticmethod
    def _build_formatter(anonymise: bool) -> logging.Formatter:
        """Build a formatter that hides personally identifiable data when requested."""

        if anonymise:
            format_string = "%(asctime)s | %(levelname)s | %(component)s | %(message)s"
        else:
            format_string = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

        return _ComponentSafeFormatter(format_string)
