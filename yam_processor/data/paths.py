"""Filesystem path sanitisation helpers for user-supplied locations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PyQt5 import QtCore  # type: ignore

_ALLOWED_ROOTS: tuple[Path, ...] = ()


def configure_allowed_roots(roots: Iterable[Path]) -> None:
    """Configure the directories that are considered safe for user paths."""

    resolved: list[Path] = []
    for root in roots:
        path = Path(root).expanduser()
        try:
            resolved_root = path.resolve(strict=True)
        except FileNotFoundError:
            # Allow configuration of directories that will be created later.
            resolved_root = path.resolve(strict=False)
        if resolved_root.exists() and resolved_root.is_symlink():
            raise ValueError(f"Allowed root may not be a symlink: {resolved_root}")
        if not resolved_root.exists():
            resolved_root.mkdir(parents=True, exist_ok=True)
        resolved.append(resolved_root)
    global _ALLOWED_ROOTS
    _ALLOWED_ROOTS = tuple(resolved)


def allowed_roots() -> tuple[Path, ...]:
    """Return the currently configured allowed roots."""

    if not _ALLOWED_ROOTS:
        configure_allowed_roots([Path.cwd()])
    return _ALLOWED_ROOTS


def _raise_error(message: str) -> None:
    translated = QtCore.QCoreApplication.translate("PathSanitiser", message)
    raise ValueError(translated)


def _ensure_roots_configured() -> tuple[Path, ...]:
    roots = allowed_roots()
    if not roots:
        _raise_error("No allowed roots have been configured")
    return roots


def _contains_symlink(path: Path) -> bool:
    for ancestor in (path, *path.parents):
        if ancestor.exists() and ancestor.is_symlink():
            return True
    return False


def sanitize_user_path(
    path: os.PathLike[str] | str,
    *,
    must_exist: bool = False,
    allow_directory: bool = True,
    allow_file: bool = True,
) -> Path:
    """Normalise ``path`` ensuring it resides within the allowed roots."""

    if not allow_directory and not allow_file:
        _raise_error("Either directories or files must be permitted")

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    if _contains_symlink(candidate):
        _raise_error("Paths containing symbolic links are not permitted")

    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        _raise_error(f"Path does not exist: {candidate}")
    except RuntimeError as exc:  # pragma: no cover - defensive guard for symlink loops
        _raise_error(f"Unable to resolve path: {candidate} ({exc})")

    roots = _ensure_roots_configured()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        _raise_error("Path escapes the configured sandbox")

    if resolved.exists():
        if resolved.is_dir() and not allow_directory:
            _raise_error("A directory path was supplied where files are required")
        if resolved.is_file() and not allow_file:
            _raise_error("A file path was supplied where directories are required")

    return resolved


__all__ = [
    "allowed_roots",
    "configure_allowed_roots",
    "sanitize_user_path",
]
