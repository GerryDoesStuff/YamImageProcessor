"""Utilities for managing ordered image processing pipelines."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple, Union

import numpy as np

from .tiled_records import TiledPipelineImage
from core.tiled_image import TileBox

try:  # pragma: no cover - defensive fallback when numpy stubs lack ndarray
    NDArray = np.ndarray
except AttributeError:  # pragma: no cover - executed in minimal test environments
    NDArray = type(None)


PipelineImage = Union[NDArray, TiledPipelineImage]

LOGGER = logging.getLogger(__name__)


@dataclass
class StepExecutionMetadata:
    """Hints influencing how a :class:`PipelineStep` should be executed."""

    supports_inplace: bool = False
    requires_gpu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_inplace": self.supports_inplace,
            "requires_gpu": self.requires_gpu,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepExecutionMetadata":
        return cls(
            supports_inplace=bool(data.get("supports_inplace", False)),
            requires_gpu=bool(data.get("requires_gpu", False)),
        )

    def is_default(self) -> bool:
        return not (self.supports_inplace or self.requires_gpu)


class GpuExecutor(Protocol):
    """Protocol describing helpers capable of executing GPU steps."""

    def execute(self, step: "PipelineStep", image: NDArray) -> NDArray:
        """Execute ``step`` using an accelerator backend."""


@dataclass
class PipelineStep:
    """A single step in an image processing pipeline."""

    name: str
    function: Callable[..., PipelineImage]
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    execution: StepExecutionMetadata = field(default_factory=StepExecutionMetadata)
    supports_tiled_input: bool = False

    def apply(self, image: PipelineImage) -> PipelineImage:
        """Execute the processing step if enabled."""

        if not self.enabled:
            return image
        operand = image
        if isinstance(image, TiledPipelineImage) and not self.supports_tiled_input:
            operand = image.to_array()

        result = self.function(operand, **self.params)
        if result is None:
            result = operand
        if self.execution.supports_inplace:
            if isinstance(operand, NDArray) and isinstance(result, NDArray):
                if result is operand:
                    return operand
                if result.shape == operand.shape and result.dtype == operand.dtype:
                    operand[...] = result
                    return operand
        return result

    def clone(self) -> "PipelineStep":
        """Return a deep copy of the step."""

        return PipelineStep(
            name=self.name,
            function=self.function,
            enabled=self.enabled,
            params=dict(self.params),
            execution=StepExecutionMetadata(
                supports_inplace=self.execution.supports_inplace,
                requires_gpu=self.execution.requires_gpu,
            ),
            supports_tiled_input=self.supports_tiled_input,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "enabled": self.enabled,
            "params": dict(self.params),
        }
        if not self.execution.is_default():
            payload["execution"] = self.execution.to_dict()
        if self.supports_tiled_input:
            payload["supports_tiled_input"] = True
        return payload

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        function: Callable[..., PipelineImage],
    ) -> "PipelineStep":
        return cls(
            name=data["name"],
            function=function,
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params", {})),
            execution=StepExecutionMetadata.from_dict(data.get("execution", {})),
            supports_tiled_input=bool(data.get("supports_tiled_input", False)),
        )


@dataclass
class PipelineState:
    """Snapshot of the pipeline configuration stored in history."""

    steps: List[PipelineStep]
    image: Optional[np.ndarray] = None
    cache_signature: Optional[str] = None

    def clone(self) -> "PipelineState":
        return PipelineState(
            [step.clone() for step in self.steps],
            None if self.image is None else self.image.copy(),
            self.cache_signature,
        )


class PipelineManager:
    """Store an ordered collection of pipeline steps with undo/redo support."""

    _DEFAULT_CACHE_DIR: Optional[Path] = None
    _DEFAULT_RECOVERY_ROOT: Optional[Path] = None

    def __init__(
        self,
        steps: Optional[Iterable[PipelineStep]] = None,
        *,
        cache_dir: Optional[os.PathLike[str] | str] = None,
        recovery_root: Optional[os.PathLike[str] | str] = None,
        gpu_executor: Optional[GpuExecutor] = None,
    ) -> None:
        template = [step.clone() for step in steps or []]
        self._template: List[PipelineStep] = template
        self._steps: List[PipelineStep] = [step.clone() for step in template]
        self._undo_stack: List[PipelineState] = []
        self._redo_stack: List[PipelineState] = []
        self._cache_directory: Optional[Path] = None
        self._recovery_root: Optional[Path] = None
        self._gpu_executor: Optional[GpuExecutor] = gpu_executor
        self.set_cache_directory(cache_dir if cache_dir is not None else self._DEFAULT_CACHE_DIR)
        self.set_recovery_root(
            recovery_root if recovery_root is not None else self._DEFAULT_RECOVERY_ROOT
        )

    @classmethod
    def set_default_cache_directory(cls, path: Optional[os.PathLike[str] | str]) -> None:
        cls._DEFAULT_CACHE_DIR = None if path is None else Path(path)
        if cls._DEFAULT_CACHE_DIR is not None:
            cls._DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_default_recovery_root(cls, path: Optional[os.PathLike[str] | str]) -> None:
        cls._DEFAULT_RECOVERY_ROOT = None if path is None else Path(path)
        if cls._DEFAULT_RECOVERY_ROOT is not None:
            cls._DEFAULT_RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)

    @property
    def cache_directory(self) -> Optional[Path]:
        return self._cache_directory

    @property
    def recovery_root(self) -> Optional[Path]:
        return self._recovery_root

    def set_cache_directory(self, path: Optional[os.PathLike[str] | str]) -> None:
        directory = None if path is None else Path(path)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        self._cache_directory = directory

    def set_recovery_root(self, path: Optional[os.PathLike[str] | str]) -> None:
        base = None if path is None else Path(path)
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
        self._recovery_root = base

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[PipelineStep]:
        return iter(self._steps)

    @property
    def steps(self) -> Tuple[PipelineStep, ...]:
        return tuple(self._steps)

    def iter_enabled_steps(self) -> Iterator[PipelineStep]:
        for step in self._steps:
            if step.enabled:
                yield step

    def clone(self) -> "PipelineManager":
        clone = PipelineManager(
            self._template,
            cache_dir=self._cache_directory,
            recovery_root=self._recovery_root,
            gpu_executor=self._gpu_executor,
        )
        clone._steps = [step.clone() for step in self._steps]
        return clone

    def template_steps(self) -> Tuple[PipelineStep, ...]:
        return tuple(step.clone() for step in self._template)

    def reset(self) -> None:
        self._steps = [step.clone() for step in self._template]
        self.clear_history()

    def clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def set_gpu_executor(self, executor: Optional[GpuExecutor]) -> None:
        """Configure the accelerator used for GPU-marked steps."""

        self._gpu_executor = executor

    def replace_steps(
        self,
        steps: Iterable[PipelineStep],
        *,
        update_template: bool = False,
        preserve_history: bool = False,
    ) -> None:
        cloned = [step.clone() for step in steps]
        self._steps = cloned
        if update_template:
            self._template = [step.clone() for step in cloned]
        if not preserve_history:
            self.clear_history()

    # ------------------------------------------------------------------
    # Step manipulation
    # ------------------------------------------------------------------
    def add_step(self, step: PipelineStep, index: Optional[int] = None) -> None:
        if index is None:
            self._steps.append(step)
        else:
            self._steps.insert(index, step)

    def remove_step(self, index: int) -> PipelineStep:
        return self._steps.pop(index)

    def move_step(self, old_index: int, new_index: int) -> None:
        step = self._steps.pop(old_index)
        self._steps.insert(new_index, step)

    def swap_steps(self, index_a: int, index_b: int) -> None:
        self._steps[index_a], self._steps[index_b] = (
            self._steps[index_b],
            self._steps[index_a],
        )

    def set_order(self, order: Iterable[str]) -> None:
        order_list = list(order)
        new_steps: List[PipelineStep] = []
        lookup = {step.name: step for step in self._steps}
        for name in order_list:
            if name not in lookup:
                raise KeyError(f"Unknown pipeline step '{name}'")
            new_steps.append(lookup.pop(name))
        # Append any remaining steps preserving their relative ordering.
        for step in self._steps:
            if step.name in lookup:
                new_steps.append(step)
        self._steps = new_steps

    def get_step(self, identifier: int | str) -> PipelineStep:
        if isinstance(identifier, int):
            return self._steps[identifier]
        for step in self._steps:
            if step.name == identifier:
                return step
        raise KeyError(f"No pipeline step named '{identifier}'")

    def set_step_enabled(self, identifier: int | str, enabled: bool) -> None:
        step = self.get_step(identifier)
        step.enabled = enabled

    def toggle_step(self, identifier: int | str) -> bool:
        step = self.get_step(identifier)
        step.enabled = not step.enabled
        return step.enabled

    def update_step_params(
        self,
        identifier: int | str,
        params: Dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        step = self.get_step(identifier)
        if replace:
            step.params = dict(params)
        else:
            step.params.update(params)

    def apply(self, image: PipelineImage) -> PipelineImage:
        if isinstance(image, TiledPipelineImage):
            return self._apply_tiled(image)

        result: PipelineImage = image.copy() if isinstance(image, NDArray) else image
        for step in self.iter_enabled_steps():
            result = self._run_step(step, result)
        return result

    def _apply_tiled(self, image: TiledPipelineImage) -> PipelineImage:
        enabled_steps = list(self.iter_enabled_steps())
        if not enabled_steps:
            return image

        # If any step supports tiled inputs we fall back to the regular
        # execution path so each step can request tiles directly.
        if any(step.supports_tiled_input for step in enabled_steps):
            result: PipelineImage = image
            for step in enabled_steps:
                result = self._run_step(step, result)
            return result

        tile_size = image.tile_size
        shape = image.infer_shape()
        dtype = image.dtype or np.float32
        assembled: Optional[np.ndarray] = None

        for box, tile in image.iter_tiles(tile_size):
            tile_result: PipelineImage = np.array(tile, copy=True)
            for step in enabled_steps:
                tile_result = self._run_step(step, tile_result)
                if isinstance(tile_result, TiledPipelineImage):
                    tile_result = tile_result.to_array()
            tile_array = np.array(tile_result, copy=False)
            if assembled is None:
                dtype = tile_array.dtype
                assembled = np.zeros(shape, dtype=dtype)
            self._paste_tile(assembled, box, tile_array)

        if assembled is None:
            # No tiles were produced; return a dense copy for consistency.
            return image.to_array()
        return assembled

    @staticmethod
    def _paste_tile(target: np.ndarray, box: TileBox, tile: np.ndarray) -> None:
        left, top, right, bottom = box
        if target.ndim == 2:
            target[top:bottom, left:right] = tile
        else:
            target[top:bottom, left:right, ...] = tile

    def _run_step(self, step: PipelineStep, image: PipelineImage) -> PipelineImage:
        if step.execution.requires_gpu and self._gpu_executor is not None:
            array_input = image if isinstance(image, NDArray) else image.to_array()
            result = self._gpu_executor.execute(step, array_input)
            if result is None:
                return array_input
            return result

        if step.execution.requires_gpu and self._gpu_executor is None:
            LOGGER.warning(
                "Step '%s' requires GPU execution but no executor is configured; falling back to CPU.",
                step.name,
            )
            array_input = image if isinstance(image, NDArray) else image.to_array()
            return step.apply(array_input)
        return step.apply(image)

    # ------------------------------------------------------------------
    # History support
    # ------------------------------------------------------------------
    def _snapshot(
        self, image: Optional[np.ndarray], cache_signature: Optional[str]
    ) -> PipelineState:
        return PipelineState(
            [step.clone() for step in self._steps],
            None if image is None else image.copy(),
            cache_signature,
        )

    def push_state(
        self,
        *,
        image: Optional[np.ndarray] = None,
        cache_signature: Optional[str] = None,
    ) -> None:
        self._undo_stack.append(self._snapshot(image, cache_signature))
        self._redo_stack.clear()

    def undo(
        self,
        *,
        current_image: Optional[np.ndarray] = None,
        current_cache_signature: Optional[str] = None,
    ) -> Optional[PipelineState]:
        if not self._undo_stack:
            return None
        self._redo_stack.append(self._snapshot(current_image, current_cache_signature))
        previous = self._undo_stack.pop()
        self._steps = [step.clone() for step in previous.steps]
        return previous.clone()

    def redo(
        self,
        *,
        current_image: Optional[np.ndarray] = None,
        current_cache_signature: Optional[str] = None,
    ) -> Optional[PipelineState]:
        if not self._redo_stack:
            return None
        self._undo_stack.append(self._snapshot(current_image, current_cache_signature))
        next_state = self._redo_stack.pop()
        self._steps = [step.clone() for step in next_state.steps]
        return next_state.clone()

    def history_depth(self) -> Tuple[int, int]:
        return len(self._undo_stack), len(self._redo_stack)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self._steps]}


__all__ = [
    "GpuExecutor",
    "PipelineManager",
    "PipelineState",
    "PipelineStep",
    "StepExecutionMetadata",
]

