"""Processing layer abstractions for lazily tiled image data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np

from core.tiled_image import TileBox, TiledImageRecord

TileSize = Tuple[int, int]


@dataclass
class TiledPipelineImage:
    """Wrapper exposing tiled access helpers for pipeline steps.

    The processing layer operates primarily on eager :class:`numpy.ndarray`
    inputs. When the IO manager loads an image lazily it returns a
    :class:`~core.tiled_image.TiledImageRecord` handle instead. This wrapper
    records optional tiling hints and forwards :meth:`iter_tiles` and
    :meth:`read_region` calls to the underlying lazy handle so that pipeline
    steps can stream data without materialising the full frame.
    """

    handle: TiledImageRecord
    tile_size: Optional[TileSize] = None
    shape: Optional[Tuple[int, ...]] = field(default=None, repr=False)

    def close(self) -> None:
        """Release resources held by the underlying tiled record."""

        self.handle.close()

    def infer_shape(self) -> Tuple[int, ...]:
        """Return the shape of the lazily loaded image."""

        if self.shape is not None:
            return self.shape
        if self.handle.shape is not None:
            self.shape = tuple(self.handle.shape)
            return self.shape
        if self.handle.size is not None:
            width, height = self.handle.size
            self.shape = (int(height), int(width))
            return self.shape
        array = self.handle.to_array()
        self.shape = tuple(array.shape)
        return self.shape

    def iter_tiles(
        self, tile_size: Optional[TileSize] = None
    ) -> Iterator[Tuple[TileBox, np.ndarray]]:
        """Yield ``(box, pixels)`` covering the image using ``tile_size``."""

        size = tile_size if tile_size is not None else self.tile_size
        yield from self.handle.iter_tiles(size)

    def read_region(self, box: TileBox) -> np.ndarray:
        """Return the pixels defined by ``box``."""

        return self.handle.read_region(box)

    def to_array(self) -> np.ndarray:
        """Materialise the tiled image as a dense :class:`numpy.ndarray`."""

        array = self.handle.to_array()
        self.shape = tuple(array.shape)
        return array

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Return the pixel dtype if known."""

        if self.handle.dtype is not None:
            return self.handle.dtype
        array = self.handle.to_array()
        self.shape = tuple(array.shape)
        return array.dtype


__all__ = ["TileSize", "TiledPipelineImage"]

