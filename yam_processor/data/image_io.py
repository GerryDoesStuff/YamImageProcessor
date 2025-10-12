"""Image I/O helpers for loading and saving arrays with metadata.

This module supports round-tripping image pixel data and associated metadata for
common raster formats. PNG, JPEG, TIFF, and BMP files are loaded via
:mod:`Pillow` so that EXIF data, ICC colour profiles, and other per-image
attributes can be preserved. NumPy ``.npy`` arrays are handled through
:func:`numpy.load` and :func:`numpy.save` to retain dtype and shape information.

Metadata is surfaced to callers through :class:`ImageRecord`, which couples the
pixel :class:`numpy.ndarray` with a free-form ``dict`` of metadata. The loader
captures standard values such as the Pillow image format, mode, size, and the
serialised EXIF/ICC payloads where available. Callers may extend this metadata
with custom keys before saving.

Saving functions mirror the behaviour of :func:`load_image`, attempting to
reuse the previously captured metadata for a faithful round-trip. When saving
to Pillow-supported formats the EXIF and ICC payloads are re-applied if present
and any recognised ``info`` entries (for example DPI) are forwarded. ``.npy``
files store the raw array alongside dtype/shape descriptors for validation when
reloading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
from PIL import Image


_SUPPORTED_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(slots=True)
class ImageRecord:
    """Container coupling image pixel data with associated metadata."""

    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalise_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    return path


def load_image(path: Path | str) -> ImageRecord:
    """Load an image or ``.npy`` array from ``path`` into an :class:`ImageRecord`."""

    resolved = _normalise_path(path)
    suffix = resolved.suffix.lower()

    if suffix == ".npy":
        array = np.load(resolved, allow_pickle=False)
        metadata: Dict[str, Any] = {
            "format": "NPY",
            "dtype": str(array.dtype),
            "shape": array.shape,
        }
        return ImageRecord(data=array, metadata=metadata)

    if suffix not in _SUPPORTED_RASTER_SUFFIXES:
        raise ValueError(f"Unsupported image format: {resolved.suffix}")

    with Image.open(resolved) as img:
        # Pillow lazily loads pixel data; convert to ensure a concrete ndarray.
        array = np.array(img)
        metadata = {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "info": dict(img.info),
        }
        exif = img.getexif()
        if exif:
            metadata["exif"] = exif.tobytes()
        icc_profile = img.info.get("icc_profile")
        if icc_profile:
            metadata["icc_profile"] = icc_profile

    return ImageRecord(data=array, metadata=metadata)


def _prepare_save_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    save_args: Dict[str, Any] = {}

    info = metadata.get("info")
    if isinstance(info, Mapping):
        for key, value in info.items():
            if isinstance(key, str) and key not in {"exif", "icc_profile"}:
                save_args[key] = value

    exif = metadata.get("exif")
    if exif is not None:
        save_args["exif"] = exif

    icc_profile = metadata.get("icc_profile")
    if icc_profile is not None:
        save_args["icc_profile"] = icc_profile

    return save_args


def save_image(record: ImageRecord, path: Path | str, format: Optional[str] = None) -> None:
    """Persist ``record`` to ``path`` using ``format`` if supplied."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fmt = (format or destination.suffix.lstrip(".")).upper()
    if fmt == "NPY":
        np.save(destination, record.data, allow_pickle=False)
        metadata = record.metadata
        if isinstance(metadata, MutableMapping):
            metadata.setdefault("format", "NPY")
            metadata.setdefault("dtype", str(record.data.dtype))
            metadata.setdefault("shape", record.data.shape)
        return

    # For raster formats defer to Pillow for encoding.
    if destination.suffix.lower() not in _SUPPORTED_RASTER_SUFFIXES and fmt.lower() not in {
        s.lstrip(".") for s in _SUPPORTED_RASTER_SUFFIXES
    }:
        raise ValueError(f"Unsupported image format for saving: {fmt}")

    array = np.asarray(record.data)
    image = Image.fromarray(array)
    if isinstance(record.metadata, Mapping):
        target_mode = record.metadata.get("mode")
        if isinstance(target_mode, str) and target_mode != image.mode:
            try:
                image = image.convert(target_mode)
            except ValueError:
                # Ignore invalid conversions and keep the Pillow-derived mode.
                pass
        save_kwargs = _prepare_save_metadata(record.metadata)
    else:
        save_kwargs = {}

    image.save(destination, format=format, **save_kwargs)

    if isinstance(record.metadata, MutableMapping):
        record.metadata.setdefault("format", image.format or fmt)
        record.metadata.setdefault("mode", image.mode)
        record.metadata.setdefault("size", image.size)
