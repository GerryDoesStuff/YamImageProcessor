from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

from yam_processor.processing.pipeline_manager import (
    PipelineExecutionError,
    PipelineFailure,
    PipelineManager,
    PipelineStep,
)

if "numpy" in sys.modules and not hasattr(sys.modules["numpy"], "float32"):
    del sys.modules["numpy"]
np = importlib.import_module("numpy")


def _add_value(image, *, value: float):
    return image + value


def _multiply_value(image, *, factor: float):
    return image * factor


def _explode(image):
    raise RuntimeError("kaboom")


@pytest.fixture()
def sample_image():
    assert hasattr(np, "float32")
    return np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32
    )


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
