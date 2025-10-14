"""Compatibility layer re-exporting path sanitisation helpers."""

from __future__ import annotations

from core.path_sanitizer import (  # noqa: F401 - re-exports for legacy imports
    ROOT_PLACEHOLDER,
    PathValidationError,
    allowed_roots,
    configure_allowed_roots,
    redact_path_for_metadata,
    root_index_for_path,
    sanitize_user_path,
)

__all__ = [
    "ROOT_PLACEHOLDER",
    "PathValidationError",
    "allowed_roots",
    "configure_allowed_roots",
    "redact_path_for_metadata",
    "root_index_for_path",
    "sanitize_user_path",
]

