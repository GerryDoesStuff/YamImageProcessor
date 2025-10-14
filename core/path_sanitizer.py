"""Filesystem path sanitisation helpers used throughout the application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

ROOT_PLACEHOLDER = "[root]"

_ALLOWED_ROOTS: tuple[Path, ...] = ()


class PathValidationError(ValueError):
    """Raised when a user-supplied path cannot be accepted."""


def _normalise_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        raise PathValidationError(f"Unable to resolve path '{candidate}': {exc}") from exc
    return resolved


def _deduplicate_roots(roots: Sequence[Path]) -> tuple[Path, ...]:
    seen = set()
    ordered: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        ordered.append(root)
    return tuple(ordered)


def configure_allowed_roots(roots: Iterable[Path | str]) -> None:
    """Configure the directories that are considered safe for user paths."""

    resolved: list[Path] = []
    for root in roots:
        if root is None:
            continue
        candidate = _normalise_path(root)
        if candidate.exists() and candidate.is_symlink():
            raise PathValidationError(
                f"Allowed root may not be a symbolic link: {candidate}"
            )
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
        resolved.append(candidate)

    if not resolved:
        resolved.append(_normalise_path(Path.cwd()))

    global _ALLOWED_ROOTS
    _ALLOWED_ROOTS = _deduplicate_roots(resolved)


def allowed_roots() -> tuple[Path, ...]:
    """Return the currently configured allowed roots."""

    if not _ALLOWED_ROOTS:
        configure_allowed_roots([Path.cwd()])
    return _ALLOWED_ROOTS


def _contains_symlink(path: Path) -> bool:
    for ancestor in _iter_ancestors(path):
        if ancestor.exists() and ancestor.is_symlink():
            return True
    return False


def _iter_ancestors(path: Path) -> Iterator[Path]:
    yield path
    yield from path.parents


def _ensure_roots_configured() -> tuple[Path, ...]:
    roots = allowed_roots()
    if not roots:
        raise PathValidationError("No allowed roots have been configured")
    return roots


def sanitize_user_path(
    path: os.PathLike[str] | str,
    *,
    must_exist: bool = False,
    allow_directory: bool = True,
    allow_file: bool = True,
) -> Path:
    """Normalise ``path`` ensuring it resides within the allowed roots."""

    if not allow_directory and not allow_file:
        raise PathValidationError("Either directories or files must be permitted")

    candidate = _normalise_path(path)

    if _contains_symlink(candidate):
        raise PathValidationError("Paths containing symbolic links are not permitted")

    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise PathValidationError(f"Path does not exist: {candidate}") from exc

    roots = _ensure_roots_configured()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise PathValidationError("Path escapes the configured sandbox")

    if resolved.exists():
        if resolved.is_dir() and not allow_directory:
            raise PathValidationError(
                "A directory path was supplied where files are required"
            )
        if resolved.is_file() and not allow_file:
            raise PathValidationError(
                "A file path was supplied where directories are required"
            )

    return resolved


def root_index_for_path(path: Path) -> Optional[int]:
    """Return the index of the allowed root containing ``path`` if available."""

    resolved = Path(path)
    for index, root in enumerate(allowed_roots()):
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return index
    return None


def redact_path_for_metadata(path: Path) -> str:
    """Return a redacted representation of ``path`` suitable for metadata."""

    resolved = Path(path)
    for root in allowed_roots():
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return str(Path(ROOT_PLACEHOLDER) / relative)
    return resolved.name or str(resolved)


__all__ = [
    "ROOT_PLACEHOLDER",
    "PathValidationError",
    "allowed_roots",
    "configure_allowed_roots",
    "redact_path_for_metadata",
    "root_index_for_path",
    "sanitize_user_path",
]

