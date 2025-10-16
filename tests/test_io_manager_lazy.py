from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from core.io_manager import IOManager, TiledImageRecord
from core.path_sanitizer import configure_allowed_roots


@pytest.fixture()
def io_manager() -> IOManager:
    return IOManager({})


@pytest.fixture(autouse=True)
def _configure_roots(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path])


def _write_bgr_image(path: Path, array: np.ndarray) -> None:
    if not cv2.imwrite(str(path), array):
        raise RuntimeError("Failed to write test image")


def test_lazy_load_png_returns_tiled_record(tmp_path: Path, io_manager: IOManager) -> None:
    array = np.zeros((6, 8, 3), dtype=np.uint8)
    array[..., 0] = 255  # Blue channel for BGR ordering
    path = tmp_path / "sample.png"
    _write_bgr_image(path, array)

    record, metadata = io_manager.load_image(path, lazy=True)

    assert isinstance(record, TiledImageRecord)
    assert metadata is None

    tile = record.read_region((0, 0, 4, 3))
    assert tile.shape == (3, 4, 3)
    np.testing.assert_array_equal(tile, array[:3, :4])
    np.testing.assert_array_equal(record.to_array(), array)
    record.close()


def test_lazy_load_preserves_metadata(tmp_path: Path, io_manager: IOManager) -> None:
    array = np.full((4, 4), 42, dtype=np.uint8)
    path = tmp_path / "meta.png"
    _write_bgr_image(path, array)

    metadata = {"author": "tester"}
    save_result = io_manager.save_image(path, array, metadata=metadata)

    record, loaded_metadata = io_manager.load_image(save_result.image_path, lazy=True)

    assert isinstance(record, TiledImageRecord)
    assert loaded_metadata is not None
    assert loaded_metadata.get("metadata", {}).get("author") == "tester"
    assert loaded_metadata.get("image", {}).get("format") == "PNG"
    record.close()


def test_lazy_load_npy_returns_memmap_record(
    tmp_path: Path, io_manager: IOManager
) -> None:
    array = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = tmp_path / "sample.npy"
    np.save(path, array, allow_pickle=False)

    record, metadata = io_manager.load_image(path, lazy=True)

    assert isinstance(record, TiledImageRecord)
    assert metadata is None
    np.testing.assert_array_equal(record.read_region((2, 2, 6, 5)), array[2:5, 2:6])
    np.testing.assert_array_equal(record.to_array(), array)
    assert record.dtype == array.dtype
    record.close()
