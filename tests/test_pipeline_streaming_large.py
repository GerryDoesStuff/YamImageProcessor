from __future__ import annotations

import time
from pathlib import Path
import sys
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.pipeline_cache import PipelineCache
from processing.pipeline_manager import PipelineManager, PipelineStep
from processing.tiled_records import TiledPipelineImage

TileBox = Tuple[int, int, int, int]


def _expected_tile_sequence(
    width: int, height: int, tile_size: Optional[Tuple[int, int]]
) -> List[TileBox]:
    if tile_size is None:
        return [(0, 0, width, height)]
    tile_w, tile_h = tile_size
    boxes: List[TileBox] = []
    for top in range(0, height, tile_h):
        bottom = min(top + tile_h, height)
        for left in range(0, width, tile_w):
            right = min(left + tile_w, width)
            boxes.append((left, top, right, bottom))
    return boxes


def _generate_region(
    width: int,
    height: int,
    channels: int,
    dtype: np.dtype,
    box: TileBox,
) -> np.ndarray:
    left, top, right, bottom = box
    y = np.arange(top, bottom, dtype=dtype)[:, None]
    x = np.arange(left, right, dtype=dtype)[None, :]
    base = y * width + x
    if channels == 1:
        return base.astype(dtype, copy=False)
    planes = [base + offset for offset in range(channels)]
    return np.stack(planes, axis=-1).astype(dtype, copy=False)


class _SyntheticStreamingRecord:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        channels: int,
        tile_size: Tuple[int, int],
        dtype: np.dtype = np.float32,
    ) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self.tile_size = tile_size
        self.dtype = np.dtype(dtype)
        if channels == 1:
            self.shape = (height, width)
        else:
            self.shape = (height, width, channels)
        self.size = (width, height)
        self._tiles_requested: List[TileBox] = []
        self._iteration_sizes: List[Optional[Tuple[int, int]]] = []
        self._to_array_called = False

    def close(self) -> None:  # pragma: no cover - compatibility stub
        pass

    @property
    def tiles_requested(self) -> List[TileBox]:
        return list(self._tiles_requested)

    @property
    def iteration_sizes(self) -> List[Optional[Tuple[int, int]]]:
        return list(self._iteration_sizes)

    @property
    def to_array_called(self) -> bool:
        return self._to_array_called

    def iter_tiles(
        self, tile_size: Optional[Tuple[int, int]] = None
    ) -> Iterator[Tuple[TileBox, np.ndarray]]:
        size = tile_size if tile_size is not None else self.tile_size
        self._iteration_sizes.append(size)
        boxes = _expected_tile_sequence(self.width, self.height, size)
        for box in boxes:
            self._tiles_requested.append(box)
            yield box, _generate_region(
                self.width, self.height, self.channels, self.dtype, box
            )

    def read_region(self, box: TileBox) -> np.ndarray:
        return _generate_region(self.width, self.height, self.channels, self.dtype, box)

    def to_array(self) -> np.ndarray:  # pragma: no cover - should never be called
        self._to_array_called = True
        raise AssertionError("Streaming record should not be materialised")


def test_pipeline_manager_streams_large_records_sequentially() -> None:
    record = _SyntheticStreamingRecord(
        width=1536, height=1024, channels=2, tile_size=(256, 256)
    )
    tiled = TiledPipelineImage(record, tile_size=record.tile_size, shape=record.shape)
    steps = (
        PipelineStep("offset", lambda arr, *, value: arr + value, params={"value": 4.0}),
        PipelineStep("scale", lambda arr, *, factor: arr * factor, params={"factor": 0.5}),
    )
    manager = PipelineManager(steps)

    result = manager.apply(tiled)

    assert isinstance(result, np.ndarray)
    expected_source = _generate_region(
        record.width, record.height, record.channels, record.dtype, (0, 0, record.width, record.height)
    )
    np.testing.assert_allclose(result, (expected_source + 4.0) * 0.5)
    assert not record.to_array_called
    expected_sequence = _expected_tile_sequence(
        record.width, record.height, record.tile_size
    )
    assert record.tiles_requested == expected_sequence


def test_pipeline_cache_streaming_preserves_tile_iteration(tmp_path: Path) -> None:
    record = _SyntheticStreamingRecord(
        width=960, height=720, channels=1, tile_size=(192, 180)
    )
    tiled = TiledPipelineImage(record, tile_size=record.tile_size, shape=record.shape)
    cache = PipelineCache(cache_directory=tmp_path)

    source_array = _generate_region(
        record.width, record.height, record.channels, record.dtype, (0, 0, record.width, record.height)
    )
    source_id = cache.register_source(source_array)

    steps = (
        PipelineStep("bias", lambda arr, *, delta: arr + delta, params={"delta": 2.0}),
        PipelineStep("square", lambda arr: arr * arr),
    )

    result = cache.compute(source_id, tiled, steps)

    expected_sequence = _expected_tile_sequence(
        record.width, record.height, record.tile_size
    )
    assert record.tiles_requested == expected_sequence
    assert record.iteration_sizes == [record.tile_size]
    assert not record.to_array_called

    expected = (source_array + 2.0) ** 2
    np.testing.assert_allclose(result.image, expected)


@pytest.mark.performance
def test_streaming_pipeline_performance_budget(tmp_path: Path) -> None:
    resource = pytest.importorskip("resource")

    record = _SyntheticStreamingRecord(
        width=2048, height=1536, channels=1, tile_size=(256, 256)
    )
    tiled = TiledPipelineImage(record, tile_size=record.tile_size, shape=record.shape)
    manager = PipelineManager(
        (
            PipelineStep("offset", lambda arr, *, value: arr + value, params={"value": 1.0}),
            PipelineStep("gain", lambda arr, *, gain: arr * gain, params={"gain": 1.2}),
        )
    )

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    result = manager.apply(tiled)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert isinstance(result, np.ndarray)

    expected_source = _generate_region(
        record.width, record.height, record.channels, record.dtype, (0, 0, record.width, record.height)
    )
    np.testing.assert_allclose(result, (expected_source + 1.0) * 1.2)

    delta_kb = max(0, after - before)
    expected_kb = result.nbytes / 1024
    assert elapsed < 3.0
    assert delta_kb <= max(expected_kb * 4, 512_000)
    assert not record.to_array_called
