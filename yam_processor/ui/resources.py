"""UI resource helpers for Yam Image Processor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PyQt5 import QtCore, QtGui  # type: ignore

_ICON_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0)


def _icons_directory() -> Path:
    return Path(__file__).with_name("icons")


def _resolve_icon_path(name: str) -> Path:
    candidate = _icons_directory() / f"{name}.svg"
    if not candidate.exists():
        raise FileNotFoundError(f"Icon '{name}' not found at {candidate}")
    return candidate


@lru_cache(maxsize=64)
def load_icon(name: str, base_size: int = 24) -> QtGui.QIcon:
    """Return a high-DPI aware :class:`QtGui.QIcon` for ``name``.

    The icon is generated from an SVG asset and scaled for a variety of
    device pixel ratios so that Qt can choose an appropriate pixmap when the
    DPI changes.
    """

    icon_path = _resolve_icon_path(name)
    icon = QtGui.QIcon()
    for scale in _ICON_SCALES:
        size = int(base_size * scale)
        icon.addFile(str(icon_path), QtCore.QSize(size, size))
    return icon


__all__ = ["load_icon"]
