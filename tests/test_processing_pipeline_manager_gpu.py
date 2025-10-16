from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - defensive for test runner
    sys.path.insert(0, str(ROOT))

from processing.pipeline_manager import (
    PipelineManager,
    PipelineStep,
    StepExecutionMetadata,
)
from plugins.module_base import ModuleBase, ModuleMetadata, ModuleStage


np = pytest.importorskip("numpy")


def _add(image: np.ndarray, *, value: float) -> np.ndarray:
    return image + value


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, step: PipelineStep, image: np.ndarray) -> np.ndarray:
        self.calls.append(step.name)
        return step.function(image, **step.params)


@pytest.fixture()
def sample_image() -> np.ndarray:
    return np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)


def test_pipeline_step_metadata_persistence() -> None:
    step = PipelineStep(
        "add",
        _add,
        params={"value": 2.0},
        execution=StepExecutionMetadata(supports_inplace=True, requires_gpu=True),
    )

    clone = step.clone()
    assert clone.execution == step.execution
    assert clone.execution is not step.execution

    payload = step.to_dict()
    assert payload["execution"] == {"supports_inplace": True, "requires_gpu": True}

    restored = PipelineStep.from_dict(payload, _add)
    assert restored.execution.requires_gpu is True
    assert restored.execution.supports_inplace is True


def test_gpu_executor_used_for_marked_steps(sample_image: np.ndarray) -> None:
    executor = _RecordingExecutor()
    manager = PipelineManager(
        [
            PipelineStep(
                "gpu_add",
                _add,
                params={"value": 1.0},
                execution=StepExecutionMetadata(requires_gpu=True),
            )
        ],
        gpu_executor=executor,
    )

    output = manager.apply(sample_image)

    assert executor.calls == ["gpu_add"]
    np.testing.assert_allclose(output, sample_image + 1.0)


def test_gpu_only_step_warns_without_executor(sample_image: np.ndarray, caplog: pytest.LogCaptureFixture) -> None:
    manager = PipelineManager(
        [
            PipelineStep(
                "gpu_add",
                _add,
                params={"value": 1.0},
                execution=StepExecutionMetadata(requires_gpu=True),
            )
        ]
    )

    with caplog.at_level("WARNING"):
        output = manager.apply(sample_image)

    assert "requires GPU execution" in caplog.text
    np.testing.assert_allclose(output, sample_image + 1.0)


class _GpuModule(ModuleBase):
    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="gpu-module",
            title="GPU Module",
            stage=ModuleStage.PREPROCESSING,
        )

    def pipeline_execution_metadata(self) -> StepExecutionMetadata:
        return StepExecutionMetadata(requires_gpu=True)

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover - trivial
        return image


def test_module_declares_gpu_requirement() -> None:
    module = _GpuModule()
    step = module.create_pipeline_step()
    assert step.execution.requires_gpu is True
