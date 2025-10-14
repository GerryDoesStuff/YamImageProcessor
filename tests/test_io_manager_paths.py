"""Integration tests ensuring IOManager respects path sanitisation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.io_manager import IOManager, PersistenceError
from core.path_sanitizer import ROOT_PLACEHOLDER, configure_allowed_roots


def _io_manager() -> IOManager:
    return IOManager({})


def test_save_image_within_allowed_root(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path])
    manager = _io_manager()
    destination = tmp_path / "output" / "sample.png"
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result = manager.save_image(destination, image)

    assert result.image_path == destination
    metadata_payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["image"]["path"] == str(destination)
    assert metadata_payload["image"]["display_path"].startswith(ROOT_PLACEHOLDER)


def test_save_image_outside_allowed_root_raises(tmp_path: Path) -> None:
    allowed = tmp_path / "sandbox"
    allowed.mkdir()
    configure_allowed_roots([allowed])
    manager = _io_manager()
    outside = tmp_path / "other" / "image.png"
    outside.parent.mkdir()
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(PersistenceError):
        manager.save_image(outside, image)


def test_load_image_rejects_outside_root(tmp_path: Path) -> None:
    allowed = tmp_path / "sandbox"
    allowed.mkdir()
    configure_allowed_roots([allowed])
    manager = _io_manager()
    outside = tmp_path / "other" / "array.npy"
    outside.parent.mkdir()
    np.save(outside, np.zeros((1,), dtype=np.uint8))

    with pytest.raises(PersistenceError):
        manager.load_image(outside)


def test_load_image_within_root(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path])
    manager = _io_manager()
    target = tmp_path / "data" / "array.npy"
    target.parent.mkdir()
    array = np.arange(6, dtype=np.int16)
    np.save(target, array)

    loaded, metadata = manager.load_image(target)

    assert np.array_equal(loaded, array)
    assert metadata is None
