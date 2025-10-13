"""Settings manager built on top of QSettings with JSON import/export support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - import handling logic
    from PyQt5.QtCore import QSettings  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PySide6.QtCore import QSettings  # type: ignore
    except Exception:  # pragma: no cover
        QSettings = None  # type: ignore


class _FallbackSettings:
    """A tiny in-memory substitute for QSettings when Qt is unavailable."""

    def __init__(self, organization: str, application: str) -> None:
        self._organization = organization
        self._application = application
        self._store: Dict[str, Any] = {}

    def setValue(self, key: str, value: Any) -> None:
        self._store[key] = value

    def value(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def remove(self, key: str) -> None:
        self._store.pop(key, None)

    def allKeys(self) -> Any:
        return list(self._store.keys())

    def contains(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def sync(self) -> None:  # pragma: no cover - compatibility shim
        return None


class SettingsManager:
    """High level interface around QSettings supporting JSON serialisation."""

    def __init__(self, organization: str, application: str) -> None:
        backend = QSettings if QSettings is not None else _FallbackSettings
        self._settings = backend(organization, application)
        self.organization = organization
        self.application = application

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._settings.value(key, default)

    def set(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)
        self._settings.sync()

    def remove(self, key: str) -> None:
        self._settings.remove(key)
        self._settings.sync()

    def clear(self) -> None:
        self._settings.clear()
        self._settings.sync()

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in self._all_keys():
            result[key] = self._settings.value(key)
        return result

    def from_dict(self, values: Dict[str, Any], *, clear: bool = False) -> None:
        if clear:
            self.clear()
        for key, value in values.items():
            self._settings.setValue(key, value)
        self._settings.sync()

    @property
    def backend(self) -> Any:
        """Return the underlying :class:`QSettings` compatible object."""

        return self._settings

    def export_json(self, path: Path) -> None:
        path = Path(path)
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)

    def import_json(self, path: Path, *, clear: bool = False) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Settings JSON must describe an object")
        self.from_dict(data, clear=clear)

    def _all_keys(self) -> List[str]:
        if hasattr(self._settings, "allKeys"):
            return list(self._settings.allKeys())
        raise AttributeError("Settings backend does not support allKeys")
