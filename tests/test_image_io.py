"""Unit tests for the :mod:`yam_processor.data.image_io` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip(
    "PIL.Image", reason="Pillow is required for image tests"
)

from yam_processor.data import image_io
from core.path_sanitizer import configure_allowed_roots


@pytest.fixture(autouse=True)
def _reset_lazy_threshold() -> None:
    """Ensure threshold changes in one test do not leak into another."""

    original_threshold = image_io._LAZY_PIXEL_THRESHOLD
    yield
    image_io._LAZY_PIXEL_THRESHOLD = original_threshold


@pytest.fixture(autouse=True)
def _allow_tmp_root(tmp_path: Path) -> None:
    """Ensure temporary directories are accepted by the path sanitizer."""

    configure_allowed_roots([tmp_path])


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
    # ``read_region`` and ``to_array`` both materialise pixels exactly once.
    assert call_counter["count"] == 2
    np.testing.assert_array_equal(record.to_array(), array)
    assert call_counter["count"] == 2
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


def test_load_multipage_tiff_returns_dimensional_record(tmp_path: Path) -> None:
    frames = [np.full((5, 5), fill_value=index, dtype=np.uint16) for index in range(4)]
    path = tmp_path / "volume.tiff"
    pil_frames = [PIL_Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:])

    record = image_io.load_image(path)

    assert isinstance(record, image_io.DimensionalImageRecord)
    assert record.dims == ("z", "y", "x")
    assert record.metadata["frames"] == len(frames)
    np.testing.assert_array_equal(record.to_array()[2], frames[2])


def test_save_and_load_npz_preserves_dims(tmp_path: Path) -> None:
    array = np.random.randint(0, 255, size=(2, 3, 4, 5), dtype=np.uint8)
    record = image_io.DimensionalImageRecord(data=array, dims=("t", "z", "y", "x"))
    path = tmp_path / "cube.npz"

    image_io.save_image(record, path)

    loaded = image_io.load_image(path)
    assert isinstance(loaded, image_io.DimensionalImageRecord)
    assert tuple(loaded.dims) == ("t", "z", "y", "x")
    np.testing.assert_array_equal(loaded.to_array(), array)


def test_save_and_load_hdf5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    array = np.random.rand(3, 4, 5).astype(np.float32)
    record = image_io.DimensionalImageRecord(data=array, dims=("z", "y", "x"))
    path = tmp_path / "volume.h5"

    image_io.save_image(record, path)

    loaded = image_io.load_image(path)
    assert isinstance(loaded, image_io.DimensionalImageRecord)
    assert loaded.metadata["format"] == "HDF5"
    assert tuple(loaded.dims) == ("z", "y", "x")
    np.testing.assert_allclose(loaded.to_array(), array)
