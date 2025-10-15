"""Unit tests for the :mod:`yam_processor.data.image_io` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip(
    "PIL.Image", reason="Pillow is required for image tests"
)

from yam_processor.data import image_io


@pytest.fixture(autouse=True)
def _reset_lazy_threshold() -> None:
    """Ensure threshold changes in one test do not leak into another."""

    original_threshold = image_io._LAZY_PIXEL_THRESHOLD
    yield
    image_io._LAZY_PIXEL_THRESHOLD = original_threshold


def test_load_image_small_returns_eager_record(tmp_path: Path) -> None:
    array = np.arange(9, dtype=np.uint8).reshape(3, 3)
    path = tmp_path / "small.png"
    PIL_Image.fromarray(array).save(path)

    record = image_io.load_image(path)

    assert isinstance(record, image_io.ImageRecord)
    np.testing.assert_array_equal(record.data, array)
    assert record.metadata["size"] == (3, 3)


def test_load_image_large_returns_tiled_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    array = np.zeros((4, 4), dtype=np.uint8)
    path = tmp_path / "large.png"
    PIL_Image.fromarray(array).save(path)

    monkeypatch.setattr(image_io, "_LAZY_PIXEL_THRESHOLD", 4)
    call_counter = {"count": 0}
    real_image_to_array = image_io._image_to_array

    def _tracking_convert(img):
        call_counter["count"] += 1
        return real_image_to_array(img)

    monkeypatch.setattr(image_io, "_image_to_array", _tracking_convert)

    record = image_io.load_image(path)

    assert isinstance(record, image_io.TiledImageRecord)
    assert call_counter["count"] == 0

    tile = record.read_region((0, 0, 2, 2))
    assert tile.shape == (2, 2)
    np.testing.assert_array_equal(record.to_array(), array)
    assert call_counter["count"] == 1
    record.close()


def test_tiled_iter_tiles_stride(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    array = np.arange(16, dtype=np.uint8).reshape(4, 4)
    path = tmp_path / "grid.png"
    PIL_Image.fromarray(array).save(path)

    monkeypatch.setattr(image_io, "_LAZY_PIXEL_THRESHOLD", 4)
    record = image_io.load_image(path)

    tiles = list(record.iter_tiles((2, 2)))
    assert len(tiles) == 4
    for (left, top, right, bottom), data in tiles:
        np.testing.assert_array_equal(data, array[top:bottom, left:right])

    # Full materialisation remains available for compatibility.
    np.testing.assert_array_equal(record.to_array(), array)
    record.close()
