"""Data layer utilities for image file input and output."""

from .image_io import ImageRecord, TiledImageRecord, load_image, save_image
from .paths import (
    ROOT_PLACEHOLDER,
    PathValidationError,
    allowed_roots,
    configure_allowed_roots,
    redact_path_for_metadata,
    root_index_for_path,
    sanitize_user_path,
)

__all__ = [
    "ImageRecord",
    "TiledImageRecord",
    "ROOT_PLACEHOLDER",
    "PathValidationError",
    "allowed_roots",
    "configure_allowed_roots",
    "redact_path_for_metadata",
    "root_index_for_path",
    "load_image",
    "sanitize_user_path",
    "save_image",
]
