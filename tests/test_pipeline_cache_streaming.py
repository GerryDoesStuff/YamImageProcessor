from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from processing.pipeline_cache import PipelineCache, TileCacheEntry
from processing.pipeline_manager import PipelineStep
from processing.tiled_records import TiledPipelineImage


def _add_value(image, *, value: float):
    return image + value


def _multiply_value(image, *, factor: float):
    return image * factor


class _StreamingRecord:
    def __init__(self, array: np.ndarray, tile_size: Tuple[int, int]) -> None:
        self._array = array
        self.tile_size = tile_size
        self.shape = array.shape
        self.dtype = array.dtype
        self.size = (array.shape[1], array.shape[0])

    def close(self) -> None:  # pragma: no cover - compatibility stub
        pass

    def iter_tiles(self, tile_size: Tuple[int, int] | None = None) -> Iterator[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        size = tile_size if tile_size is not None else self.tile_size
        if size is None:
            yield (0, 0, self.size[0], self.size[1]), self._array
            return
        width, height = size
        for top in range(0, self._array.shape[0], height):
            bottom = min(top + height, self._array.shape[0])
            for left in range(0, self._array.shape[1], width):
                right = min(left + width, self._array.shape[1])
                yield (left, top, right, bottom), self._array[top:bottom, left:right]

    def read_region(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = box
        return self._array[top:bottom, left:right]

    def to_array(self) -> np.ndarray:  # pragma: no cover - should not be called
        raise AssertionError("Streaming cache should not materialise the full array")


def test_pipeline_cache_streams_tiles(tmp_path: Path) -> None:
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    record = _StreamingRecord(base, tile_size=(2, 2))
    tiled = TiledPipelineImage(record, tile_size=(2, 2))

    cache = PipelineCache(cache_directory=tmp_path)
    source_id = cache.register_source(base)

    steps = (
        PipelineStep("add", _add_value, params={"value": 1.0}),
        PipelineStep("multiply", _multiply_value, params={"factor": 2.0}),
    )

    result = cache.compute(source_id, tiled, steps)
    expected = (base + 1.0) * 2.0

    np.testing.assert_allclose(result.image, expected)

    with cache._lock:  # type: ignore[attr-defined]
        entry = cache._cache[source_id][result.final_signature]
    assert isinstance(entry, TileCacheEntry)

    disk_path = cache.cache_directory / f"{source_id}_{result.final_signature}.npz"
    assert disk_path.exists()

    second = cache.compute(source_id, tiled, steps)
    np.testing.assert_allclose(second.image, expected)
