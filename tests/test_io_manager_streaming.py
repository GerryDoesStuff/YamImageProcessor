from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # pragma: no cover - import guard for optional dependency
    import cv2  # noqa: F401
except ImportError:
    pytest.skip("cv2 with OpenGL support is required for IO manager streaming tests", allow_module_level=True)

from core.io_manager import DimensionalImageRecord, IOManager, TiledImageRecord
from core.path_sanitizer import configure_allowed_roots


@pytest.fixture()
def io_manager() -> IOManager:
    return IOManager({})


@pytest.fixture(autouse=True)
def _configure_roots(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path])


def _synthetic_volume(width: int, height: int, channels: int = 3) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = (np.sin(x * np.pi) + np.cos(y * np.pi)) * 128.0
    if channels == 1:
        return base.astype(np.float32)
    planes = [base + index * 5.0 for index in range(channels)]
    return np.stack(planes, axis=-1).astype(np.float32)


def test_lazy_memmap_iter_tiles_streams_without_materialising(
    tmp_path: Path, io_manager: IOManager
) -> None:
    array = _synthetic_volume(width=960, height=640, channels=3)
    path = tmp_path / "synthetic_large.npy"
    np.save(path, array, allow_pickle=False)

    record, metadata = io_manager.load_image(path, lazy=True)

    assert isinstance(record, TiledImageRecord)
    assert metadata is None

    tile_size = (128, 160)
    assembled = np.zeros_like(array)
    tiles_seen = 0

    for box, tile in record.iter_tiles(tile_size):
        left, top, right, bottom = box
        np.testing.assert_allclose(tile, array[top:bottom, left:right, ...])
        assembled[top:bottom, left:right, ...] = tile
        tiles_seen += 1

    np.testing.assert_allclose(assembled, array)
    assert tiles_seen == ((640 + tile_size[1] - 1) // tile_size[1]) * (
        (960 + tile_size[0] - 1) // tile_size[0]
    )
    assert record._cached_array is None
    record.close()


def test_lazy_memmap_read_region_matches_source(tmp_path: Path, io_manager: IOManager) -> None:
    array = _synthetic_volume(width=512, height=768, channels=2)
    path = tmp_path / "synthetic_regions.npy"
    np.save(path, array, allow_pickle=False)

    record, metadata = io_manager.load_image(path, lazy=True)

    assert isinstance(record, TiledImageRecord)
    assert metadata is None

    regions = [
        (0, 0, 128, 128),
        (100, 50, 260, 190),
        (256, 512, 512, 768),
    ]

    for box in regions:
        left, top, right, bottom = box
        region = record.read_region(box)
        np.testing.assert_allclose(region, array[top:bottom, left:right, ...])

    assert record._cached_array is None
    record.close()


def test_save_and_load_dimensional_npz(tmp_path: Path, io_manager: IOManager) -> None:
    cube = np.random.randint(0, 255, size=(4, 8, 6), dtype=np.uint8)
    record = DimensionalImageRecord(data=cube, dims=("z", "y", "x"))
    path = tmp_path / "volume.npz"

    io_manager.save_image(path, record, metadata={"source": "synthetic"})

    loaded, metadata = io_manager.load_image(path)
    assert isinstance(loaded, DimensionalImageRecord)
    assert metadata is not None
    assert metadata["metadata"]["dims"] == ["z", "y", "x"]
    np.testing.assert_array_equal(loaded.to_array(), cube)
