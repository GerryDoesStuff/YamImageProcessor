"""Packaged Qt translation catalogues for Yam Image Processor."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = ["package_path", "iter_catalogues"]


def package_path() -> Path:
    """Return the filesystem path containing the packaged translations."""

    return Path(__file__).resolve().parent


def iter_catalogues(suffix: str = ".qm") -> Iterable[Path]:
    """Iterate over packaged catalogue files with the requested suffix."""

    root = package_path()
    yield from sorted(root.glob(f"*{suffix}"))
