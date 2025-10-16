from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import importlib.util

import pytest

from core.tiled_image import TiledImageRecord
from processing.tiled_records import TiledPipelineImage

ROOT = Path(__file__).resolve().parents[1]

if "yam_processor" not in sys.modules:
    yam_pkg = types.ModuleType("yam_processor")
    yam_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["yam_processor"] = yam_pkg
else:
    yam_pkg = sys.modules["yam_processor"]

processing_pkg = types.ModuleType("yam_processor.processing")
processing_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules["yam_processor.processing"] = processing_pkg
setattr(yam_pkg, "processing", processing_pkg)

spec = importlib.util.spec_from_file_location(
    "yam_processor.processing.pipeline_manager",
    ROOT / "yam_processor" / "processing" / "pipeline_manager.py",
)
assert spec and spec.loader is not None
pipeline_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline_module
spec.loader.exec_module(pipeline_module)
setattr(processing_pkg, "pipeline_manager", pipeline_module)

PipelineExecutionError = pipeline_module.PipelineExecutionError
PipelineFailure = pipeline_module.PipelineFailure
PipelineManager = pipeline_module.PipelineManager
PipelineStep = pipeline_module.PipelineStep

try:
    np = importlib.import_module("numpy")
except ModuleNotFoundError:
    pytest.skip("numpy is required for pipeline tests", allow_module_level=True)

if not hasattr(np, "float32"):
    pytest.skip("numpy installation is incomplete", allow_module_level=True)



def _add_value(image, *, value: float):
    return image + value


def _multiply_value(image, *, factor: float):
    return image * factor


def _explode(image):
    raise RuntimeError("kaboom")


def _tiled_passthrough(image, *, offset: float = 0.0):
    assert isinstance(image, TiledPipelineImage)
    tiles = list(image.iter_tiles(image.tile_size))
    assert tiles, "expected tiled iterator to yield at least one region"
    if offset:
        return image.to_array() + offset
    return image


@pytest.fixture()
def sample_image():
    assert hasattr(np, "float32")
    return np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32
    )


@pytest.fixture()
def tiled_pipeline_image(tmp_path: Path):
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = tmp_path / "lazy.npy"
    np.save(path, array, allow_pickle=False)
    memmap = np.load(path, mmap_mode="r", allow_pickle=False)
    record = TiledImageRecord.from_npy(path, metadata={"format": "NPY"}, memmap=memmap)
    pipeline_record = TiledPipelineImage(record, tile_size=(2, 2))
    try:
        yield pipeline_record, array
    finally:
        pipeline_record.close()


def test_pipeline_execution_history_and_redo(tmp_path: Path, sample_image) -> None:
    manager = PipelineManager(
        [
            PipelineStep("add", _add_value, params={"value": 1.5}),
            PipelineStep("multiply", _multiply_value, params={"factor": 2.0}),
        ],
        cache_dir=tmp_path / "cache",
        recovery_root=tmp_path / "recovery",
    )

    first_result = manager.apply(sample_image)
    expected_first = (sample_image + 1.5) * 2.0
    assert np.allclose(first_result, expected_first)

    manager.push_history(first_result)
    assert manager.history_depth() == (1, 0)

    manager.set_step_enabled("multiply", False)
    second_result = manager.apply(sample_image)
    assert np.allclose(second_result, sample_image + 1.5)

    undo_entry = manager.undo(second_result)
    assert undo_entry is not None
    assert manager.history_depth() == (0, 1)
    assert manager.get_step("multiply").enabled is True

    undo_output = undo_entry.get_final_output()
    assert undo_output is not None
    assert np.allclose(undo_output, expected_first)

    redo_entry = manager.redo(expected_first)
    assert redo_entry is not None
    assert manager.history_depth() == (1, 0)
    assert manager.get_step("multiply").enabled is False

    redo_output = redo_entry.get_final_output()
    assert redo_output is not None
    assert np.allclose(redo_output, second_result)


def test_pipeline_failure_records_last_failure(tmp_path: Path, sample_image) -> None:
    manager = PipelineManager(
        [
            PipelineStep("add", _add_value, params={"value": 1.0}),
            PipelineStep("explode", _explode),
        ],
        cache_dir=tmp_path / "cache",
        recovery_root=tmp_path / "recovery",
    )

    with pytest.raises(PipelineExecutionError) as exc_info:
        manager.apply(sample_image)

    failure = manager.last_failure()
    assert failure is not None
    assert failure is exc_info.value.failure
    assert isinstance(failure, PipelineFailure)
    assert failure.step_name == "explode"
    assert isinstance(failure.exception, RuntimeError)
    assert "kaboom" in str(failure.exception)
    assert failure.recovery_path.name == "traceback.txt"
    assert failure.recovery_path.exists()
    assert "RuntimeError" in failure.recovery_path.read_text(encoding="utf-8")
    assert manager.get_step("explode").enabled is False


def test_pipeline_materialises_tiled_records_when_step_lacks_support(
    tiled_pipeline_image,
) -> None:
    record, array = tiled_pipeline_image
    manager = PipelineManager([PipelineStep("add", _add_value, params={"value": 1.0})])

    result = manager.apply(record)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, array + 1.0)


def test_pipeline_preserves_tiled_records_for_supported_steps(tiled_pipeline_image) -> None:
    record, array = tiled_pipeline_image
    manager = PipelineManager(
        [PipelineStep("noop", _tiled_passthrough, supports_tiled_input=True)]
    )

    result = manager.apply(record)

    assert result is record
    np.testing.assert_allclose(record.to_array(), array)
