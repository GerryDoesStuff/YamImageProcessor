"""Lazy tiled image record for on-demand pixel access."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from PIL import Image


TileBox = Tuple[int, int, int, int]


def _iter_tile_boxes(
    width: int, height: int, tile_size: Optional[Tuple[int, int]]
) -> Iterator[TileBox]:
    if tile_size is None:
        yield (0, 0, width, height)
        return

    tile_width, tile_height = tile_size
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_size must contain positive integers")

    for top in range(0, height, tile_height):
        bottom = min(top + tile_height, height)
        for left in range(0, width, tile_width):
            right = min(left + tile_width, width)
            yield (left, top, right, bottom)


def _validate_box(box: TileBox, width: int, height: int) -> TileBox:
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            "box coordinates must define a region within the image bounds"
        )
    return left, top, right, bottom


def _rgb_to_bgr(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 3:
        return array[..., ::-1]
    if array.ndim == 3 and array.shape[2] == 4:
        bgr = array.copy()
        bgr[..., :3] = array[..., 2::-1]
        return bgr
    return array


@dataclass
class TiledImageRecord:
    """Lightweight handle around on-disk image pixels."""

    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    mode: Optional[str] = None
    size: Optional[Tuple[int, int]] = None  # (width, height)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[np.dtype] = None
    _cached_array: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _image_handle: Optional[Image.Image] = field(
        default=None, init=False, repr=False
    )
    _memmap: Optional[np.memmap] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    @classmethod
    def from_raster(
        cls, path: Path, *, metadata: Dict[str, Any], image: Image.Image
    ) -> "TiledImageRecord":
        size = image.size
        mode = image.mode
        record = cls(
            path=path,
            metadata=dict(metadata),
            mode=mode,
            size=size,
        )
        record._image_handle = image
        return record

    @classmethod
    def from_npy(
        cls, path: Path, *, metadata: Dict[str, Any], memmap: np.memmap
    ) -> "TiledImageRecord":
        record = cls(
            path=path,
            metadata=dict(metadata),
            shape=tuple(memmap.shape),
            dtype=memmap.dtype,
        )
        record._memmap = memmap
        return record

    # ------------------------------------------------------------------
    # Public API
    def close(self) -> None:
        """Release any open file handles backing this record."""

        if self._image_handle is not None:
            try:
                self._image_handle.close()
            finally:
                self._image_handle = None
        if self._memmap is not None:
            base = getattr(self._memmap, "_mmap", None)
            if base is not None:
                base.close()
            self._memmap = None

    def to_array(self) -> np.ndarray:
        """Materialise the full image as a dense array."""

        if self._cached_array is not None:
            return self._cached_array

        if self._memmap is not None:
            array = np.asarray(self._memmap)
        else:
            image = self._ensure_image()
            array = np.array(image)
            if image.mode not in {"F", "I;16"}:
                array = _rgb_to_bgr(array)
        self._cached_array = array
        if self.shape is None:
            self.shape = tuple(array.shape)
        if self.dtype is None:
            self.dtype = array.dtype
        return array

    def read_region(self, box: TileBox) -> np.ndarray:
        """Return pixels corresponding to ``box``."""

        if self._memmap is not None:
            shape = self.shape or tuple(self._memmap.shape)
            if len(shape) < 2:
                raise ValueError("np.ndarray images must be at least 2-D")
            height, width = shape[0], shape[1]
            left, top, right, bottom = _validate_box(box, width, height)
            slices = (slice(top, bottom), slice(left, right))
            if len(shape) > 2:
                slices += (slice(None),)
            return np.asarray(self._memmap[slices])

        image = self._ensure_image()
        if image.size is None:
            raise ValueError("Image size unavailable for tiled reads")
        width, height = image.size
        left, top, right, bottom = _validate_box(box, width, height)
        region = image.crop((left, top, right, bottom))
        array = np.array(region)
        if region.mode not in {"F", "I;16"}:
            array = _rgb_to_bgr(array)
        return array

    def iter_tiles(
        self, tile_size: Optional[Tuple[int, int]] = None
    ) -> Iterator[Tuple[TileBox, np.ndarray]]:
        """Yield tiles covering the full image."""

        width, height = self._infer_dimensions()
        for box in _iter_tile_boxes(width, height, tile_size):
            yield box, self.read_region(box)

    # ------------------------------------------------------------------
    # Internals
    def _ensure_image(self) -> Image.Image:
        if self._image_handle is None:
            self._image_handle = Image.open(self.path)
        return self._image_handle

    def _infer_dimensions(self) -> Tuple[int, int]:
        if self.size is not None:
            return self.size
        if self.shape is not None and len(self.shape) >= 2:
            return (int(self.shape[1]), int(self.shape[0]))
        array = self.to_array()
        if array.ndim < 2:
            raise ValueError("Cannot infer dimensions from a 1-D array")
        self.shape = tuple(array.shape)
        return (array.shape[1], array.shape[0])


__all__ = ["TiledImageRecord", "TileBox"]
