"""Top level package for the YamImageProcessor application."""

from __future__ import annotations

from importlib import metadata as _importlib_metadata
from typing import Any


def _resolve_distribution_version() -> str:
    """Best-effort retrieval of the installed package version."""

    candidates = ("yam-processor", "yam_processor", "YamImageProcessor")
    for name in candidates:
        try:
            return _importlib_metadata.version(name)
        except _importlib_metadata.PackageNotFoundError:  # pragma: no cover - metadata lookup
            continue
    return "0.0.0"


__version__ = _resolve_distribution_version()


def get_version() -> str:
    """Return the discovered package version."""

    return __version__


_QT_IMPORT_ERROR: Exception | None

try:
    from .core.app_core import AppConfiguration, AppCore
except ModuleNotFoundError as exc:  # pragma: no cover - optional Qt dependency
    if exc.name not in {"PyQt5", "PySide6"}:
        raise
    _QT_IMPORT_ERROR = exc

    def __getattr__(name: str) -> Any:  # pragma: no cover - fallback accessor
        if name in {"AppCore", "AppConfiguration"}:
            raise ModuleNotFoundError(
                "Qt bindings (PyQt5 or PySide6) are required to use "
                f"yam_processor.{name}."
            ) from _QT_IMPORT_ERROR
        raise AttributeError(name)
else:
    _QT_IMPORT_ERROR = None


__all__ = ["AppCore", "AppConfiguration", "__version__", "get_version"]

