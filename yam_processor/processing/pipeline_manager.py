"""Utilities for managing ordered image-processing pipelines."""

from __future__ import annotations

import base64
import copy
import datetime as _dt
import io
import logging
import os
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple, Union

import numpy as np

from processing.tiled_records import TiledPipelineImage, TileSize
from core.tiled_image import TileBox

try:  # pragma: no cover - defensive fallback when numpy stubs lack ndarray
    NDArray = np.ndarray
except AttributeError:  # pragma: no cover - executed in minimal test environments
    NDArray = type(None)


LOGGER = logging.getLogger(__name__)


PipelineImage = Union[NDArray, TiledPipelineImage]


@dataclass
class StepExecutionMetadata:
    """Hints controlling how a :class:`PipelineStep` should be executed.

    The ``requires_gpu`` flag is the primary signal consumed by upcoming GPU
    dispatch logic (see :mod:`docs.performance_roadmap`). Pipelines should
    continue to populate this metadata even while the CPU fallback remains in
    place so that the transition to accelerator aware execution can be driven by
    configuration rather than invasive code changes.
    """

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
    """Protocol describing GPU execution helpers.

    Implementations act as the bridge between :class:`PipelineStep` instances
    and optimised OpenCV or scikit-image kernels. The migration plan involves
    providing adapters that can accept a step and orchestrate device transfers
    before invoking the accelerated operator, falling back to the step's CPU
    ``function`` when no backend is available.
    """

    def execute(
        self,
        step: "PipelineStep",
        image: np.ndarray,
    ) -> np.ndarray:
        """Execute ``step`` on ``image`` using the accelerator."""


@dataclass
class PipelineStep:
    """Represents a single processing step within a pipeline.

    Parameters
    ----------
    name:
        Human readable identifier for the step. Used for serialisation and UI.
    function:
        Callable that accepts either an :class:`numpy.ndarray` or a
        :class:`~processing.tiled_records.TiledPipelineImage` and returns the
        processed result.
    enabled:
        Flag controlling whether the step should be executed when applying the
        pipeline.
    params:
        Keyword arguments that will be forwarded to ``function`` during
        execution. The parameters are stored so they can be serialised to disk
        and restored later.
    execution:
        Optional hints describing how the step prefers to be executed. These
        hints are serialised with the step so they can be restored across
        sessions.
    """

    name: str
    function: Callable[..., PipelineImage]
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    execution: StepExecutionMetadata = field(default_factory=StepExecutionMetadata)
    supports_tiled_input: bool = False

    def apply(self, image: PipelineImage) -> PipelineImage:
        """Execute the step against ``image`` if it is enabled."""

        if not self.enabled:
            LOGGER.debug("Skipping disabled step: %s", self.name)
            return image
        if self.execution.requires_gpu:
            LOGGER.debug(
                "Step '%s' marked for GPU execution; executing on CPU fallback", self.name
            )
        operand = image
        if isinstance(image, TiledPipelineImage) and not self.supports_tiled_input:
            LOGGER.debug(
                "Step '%s' requires dense input; materialising tiled record", self.name
            )
            operand = image.to_array()
        LOGGER.debug("Applying step '%s' with params %s", self.name, self.params)
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
        """Return a lightweight copy of the step preserving the function."""

        return PipelineStep(
            name=self.name,
            function=self.function,
            enabled=self.enabled,
            params=copy.deepcopy(self.params),
            execution=StepExecutionMetadata(
                supports_inplace=self.execution.supports_inplace,
                requires_gpu=self.execution.requires_gpu,
            ),
            supports_tiled_input=self.supports_tiled_input,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the step into a JSON friendly structure."""

        payload: Dict[str, Any] = {
            "name": self.name,
            "enabled": self.enabled,
            "params": copy.deepcopy(self.params),
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
        function_resolver: Callable[[str], Callable[..., PipelineImage]],
    ) -> "PipelineStep":
        """Create a :class:`PipelineStep` from a serialised representation.

        Parameters
        ----------
        data:
            Dictionary containing the fields generated by :meth:`to_dict`.
        function_resolver:
            Callback that accepts the step name and returns the processing
            callable. A :class:`KeyError` will be raised if the function cannot
            be resolved.
        """

        function = function_resolver(data["name"])
        return cls(
            name=data["name"],
            function=function,
            enabled=bool(data.get("enabled", True)),
            params=copy.deepcopy(data.get("params", {})),
            execution=StepExecutionMetadata.from_dict(data.get("execution", {})),
            supports_tiled_input=bool(data.get("supports_tiled_input", False)),
        )


DEFAULT_CACHE_THRESHOLD_BYTES = 1_048_576  # 1 MiB


@dataclass
class CachedArray:
    """Container for cached numpy arrays with optional disk backed storage.

    Arrays smaller than :data:`DEFAULT_CACHE_THRESHOLD_BYTES` are copied into
    memory, while larger results are serialised to ``.npy`` files on disk so
    they can be restored lazily when required.
    """

    storage: str
    _array: Optional[np.ndarray] = None
    path: Optional[str] = None

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES,
        cache_dir: Optional[os.PathLike[str] | str] = None,
    ) -> "CachedArray":
        """Create a cache entry backed by memory or disk depending on size."""

        if array.nbytes <= threshold_bytes:
            return cls("memory", array.copy(), None)
        dir_arg = os.fspath(cache_dir) if cache_dir is not None else None
        fd, filename = tempfile.mkstemp(suffix=".npy", dir=dir_arg)
        os.close(fd)
        np.save(filename, array, allow_pickle=False)
        return cls("disk", None, filename)

    @classmethod
    def from_optional(
        cls,
        array: Optional[np.ndarray],
        *,
        threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES,
        cache_dir: Optional[os.PathLike[str] | str] = None,
    ) -> Optional["CachedArray"]:
        if array is None:
            return None
        return cls.from_array(array, threshold_bytes=threshold_bytes, cache_dir=cache_dir)

    def get(self) -> np.ndarray:
        """Return a numpy array copy of the cached data."""

        if self.storage == "memory":
            assert self._array is not None
            return self._array.copy()
        if self.storage == "disk":
            assert self.path is not None
            return np.load(self.path, allow_pickle=False)
        raise ValueError(f"Unknown cache storage type '{self.storage}'")

    def clone(self) -> "CachedArray":
        if self.storage == "memory":
            assert self._array is not None
            return CachedArray("memory", self._array.copy(), None)
        return CachedArray(self.storage, None, self.path)

    def to_dict(self) -> Dict[str, Any]:
        if self.storage == "disk":
            return {"storage": "disk", "path": self.path}
        assert self._array is not None
        buffer = io.BytesIO()
        np.save(buffer, self._array, allow_pickle=False)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"storage": "inline", "payload": encoded}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedArray":
        storage = data.get("storage", "inline")
        if storage == "disk":
            return cls("disk", None, data.get("path"))
        if storage == "inline":
            payload = data.get("payload")
            if payload is None:
                raise ValueError("Inline cache payload missing")
            array = np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)
            return cls("memory", array, None)
        raise ValueError(f"Unsupported cache storage '{storage}'")


@dataclass
class PipelineHistoryEntry:
    """Snapshot of the pipeline configuration and cached outputs for undo/redo.

    The ``intermediate_outputs`` map preserves the output produced by each
    enabled step during execution, allowing downstream consumers to replay or
    inspect intermediate results without rerunning the pipeline. Large arrays
    are automatically spilled to disk using :class:`CachedArray` when they
    exceed :data:`DEFAULT_CACHE_THRESHOLD_BYTES`.
    """

    steps: List[PipelineStep]
    final_output: Optional[CachedArray]
    intermediate_outputs: Dict[str, CachedArray] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        steps: List[PipelineStep],
        output: Optional[np.ndarray],
        intermediates: Optional[Dict[str, np.ndarray]] = None,
        *,
        threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES,
        cache_dir: Optional[os.PathLike[str] | str] = None,
    ) -> "PipelineHistoryEntry":
        cached_intermediates: Dict[str, CachedArray] = {}
        if intermediates:
            for name, array in intermediates.items():
                cached_intermediates[name] = CachedArray.from_array(
                    array,
                    threshold_bytes=threshold_bytes,
                    cache_dir=cache_dir,
                )
        return cls(
            [step.clone() for step in steps],
            CachedArray.from_optional(
                output,
                threshold_bytes=threshold_bytes,
                cache_dir=cache_dir,
            ),
            cached_intermediates,
        )

    def clone(self) -> "PipelineHistoryEntry":
        return PipelineHistoryEntry(
            [step.clone() for step in self.steps],
            None if self.final_output is None else self.final_output.clone(),
            {name: cache.clone() for name, cache in self.intermediate_outputs.items()},
        )

    def get_cached_output(self, step_name: str) -> np.ndarray:
        """Return the cached output for ``step_name``."""

        if step_name not in self.intermediate_outputs:
            raise KeyError(f"No cached output available for step '{step_name}'")
        return self.intermediate_outputs[step_name].get()

    def get_final_output(self) -> Optional[np.ndarray]:
        """Return the final cached pipeline output if present."""

        if self.final_output is None:
            return None
        return self.final_output.get()

    def replay_from_cache(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield step outputs in execution order without recomputation."""

        for step in self.steps:
            cache = self.intermediate_outputs.get(step.name)
            if cache is not None:
                yield step.name, cache.get()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_output": None if self.final_output is None else self.final_output.to_dict(),
            "intermediate_outputs": {
                name: cache.to_dict() for name, cache in self.intermediate_outputs.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        function_resolver: Callable[[str], Callable[..., np.ndarray]],
    ) -> "PipelineHistoryEntry":
        steps: List[PipelineStep] = []
        for item in data.get("steps", []):
            try:
                steps.append(PipelineStep.from_dict(item, function_resolver))
            except KeyError:
                LOGGER.warning(
                    "Unknown pipeline step '%s' skipped during history load",
                    item.get("name"),
                )
        final_data = data.get("final_output")
        final_output = CachedArray.from_dict(final_data) if final_data else None
        intermediate_outputs = {
            name: CachedArray.from_dict(cache_data)
            for name, cache_data in data.get("intermediate_outputs", {}).items()
        }
        return cls(steps, final_output, intermediate_outputs)


class PipelineManager:
    """Manage an ordered collection of :class:`PipelineStep` instances.

    The manager keeps track of undo/redo history and provides helpers for
    serialising the pipeline configuration for persistence.
    """

    _DEFAULT_CACHE_DIR: Optional[str] = None
    _DEFAULT_RECOVERY_ROOT: Optional[Path] = None

    def __init__(
        self,
        steps: Optional[Iterable[PipelineStep]] = None,
        *,
        cache_dir: Optional[os.PathLike[str] | str] = None,
        recovery_root: Optional[os.PathLike[str] | str] = None,
        gpu_executor: Optional[GpuExecutor] = None,
    ) -> None:
        self.steps: List[PipelineStep] = list(steps or [])
        self._undo_stack: List[PipelineHistoryEntry] = []
        self._redo_stack: List[PipelineHistoryEntry] = []
        self._last_execution_entry: Optional[PipelineHistoryEntry] = None
        self._cache_dir = os.fspath(cache_dir) if cache_dir is not None else self._DEFAULT_CACHE_DIR
        if self._cache_dir is not None:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        recovery_base = recovery_root or self._DEFAULT_RECOVERY_ROOT
        if recovery_base is None:
            recovery_base = Path(tempfile.gettempdir()) / "yam_processor" / "recovery"
        self._recovery_root = Path(recovery_base)
        self._recovery_root.mkdir(parents=True, exist_ok=True)
        self._last_failure: Optional[PipelineFailure] = None
        self._gpu_executor: Optional[GpuExecutor] = gpu_executor

    @classmethod
    def set_default_cache_directory(cls, path: Optional[os.PathLike[str] | str]) -> None:
        cls._DEFAULT_CACHE_DIR = None if path is None else os.fspath(path)
        if cls._DEFAULT_CACHE_DIR is not None:
            Path(cls._DEFAULT_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_default_recovery_root(cls, path: Optional[os.PathLike[str] | str]) -> None:
        cls._DEFAULT_RECOVERY_ROOT = None if path is None else Path(path)
        if cls._DEFAULT_RECOVERY_ROOT is not None:
            cls._DEFAULT_RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)

    def set_cache_directory(self, path: Optional[os.PathLike[str] | str]) -> None:
        self._cache_dir = None if path is None else os.fspath(path)
        if self._cache_dir is not None:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

    def set_recovery_root(self, path: Optional[os.PathLike[str] | str]) -> None:
        base = None if path is None else Path(path)
        if base is None:
            base = Path(tempfile.gettempdir()) / "yam_processor" / "recovery"
        base.mkdir(parents=True, exist_ok=True)
        self._recovery_root = base

    def set_gpu_executor(self, executor: Optional[GpuExecutor]) -> None:
        """Register the accelerator used for GPU-only steps."""

        # TODO(gpu): Extend this setter to perform capability checks against
        # module metadata once the GPU executor exposes backend descriptors.
        self._gpu_executor = executor

    # ------------------------------------------------------------------
    # Step management helpers
    # ------------------------------------------------------------------
    def add_step(self, step: PipelineStep, index: Optional[int] = None) -> None:
        """Append ``step`` to the pipeline or insert at ``index``."""

        if index is None:
            LOGGER.debug("Appending step '%s'", step.name)
            self.steps.append(step)
        else:
            LOGGER.debug("Inserting step '%s' at index %s", step.name, index)
            self.steps.insert(index, step)

    def remove_step(self, index: int) -> PipelineStep:
        """Remove and return the step located at ``index``."""

        step = self.steps.pop(index)
        LOGGER.debug("Removed step '%s' at index %s", step.name, index)
        return step

    def move_step(self, old_index: int, new_index: int) -> None:
        """Move the step from ``old_index`` to ``new_index`` preserving order."""

        step = self.steps.pop(old_index)
        self.steps.insert(new_index, step)
        LOGGER.info(
            "Moved step '%s' from %s to %s", step.name, old_index, new_index
        )

    def swap_steps(self, index_a: int, index_b: int) -> None:
        """Swap the steps at ``index_a`` and ``index_b``."""

        self.steps[index_a], self.steps[index_b] = self.steps[index_b], self.steps[index_a]
        LOGGER.info(
            "Swapped steps '%s' and '%s'", self.steps[index_a].name, self.steps[index_b].name
        )

    def set_step_enabled(self, identifier: int | str, enabled: bool) -> None:
        """Enable or disable a step referenced by index or name."""

        step = self._resolve_step(identifier)
        if step.enabled == enabled:
            return
        step.enabled = enabled
        LOGGER.info("Step '%s' %s", step.name, "enabled" if enabled else "disabled")

    def toggle_step(self, identifier: int | str) -> None:
        step = self._resolve_step(identifier)
        step.enabled = not step.enabled
        LOGGER.info("Toggled step '%s' -> %s", step.name, step.enabled)

    def get_step(self, identifier: int | str) -> PipelineStep:
        """Return the step referenced by ``identifier`` without mutation."""

        return self._resolve_step(identifier)

    def _resolve_step(self, identifier: int | str) -> PipelineStep:
        if isinstance(identifier, int):
            return self.steps[identifier]
        for step in self.steps:
            if step.name == identifier:
                return step
        raise KeyError(f"No pipeline step named '{identifier}'")

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _snapshot_steps(self) -> List[PipelineStep]:
        return [step.clone() for step in self.steps]

    def _entry_for_current_state(self, output: Optional[np.ndarray]) -> PipelineHistoryEntry:
        if self._last_execution_entry is not None:
            entry = self._last_execution_entry.clone()
            if output is not None:
                entry.final_output = CachedArray.from_array(
                    output, cache_dir=self._cache_dir
                )
            return entry
        return PipelineHistoryEntry.from_arrays(
            self.steps,
            output,
            cache_dir=self._cache_dir,
        )

    def push_history(
        self,
        output: Optional[np.ndarray],
        intermediates: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Record the current pipeline configuration and cached outputs."""

        if intermediates is None and self._last_execution_entry is not None:
            entry = self._last_execution_entry.clone()
            if output is not None:
                entry.final_output = CachedArray.from_array(
                    output,
                    cache_dir=self._cache_dir,
                )
        else:
            entry = PipelineHistoryEntry.from_arrays(
                self.steps,
                output,
                intermediates,
                cache_dir=self._cache_dir,
            )
        self._undo_stack.append(entry.clone())
        self._redo_stack.clear()
        self._last_execution_entry = entry.clone()
        LOGGER.info("History push (undo depth=%s)", len(self._undo_stack))

    def clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        LOGGER.debug("Pipeline history cleared")

    def undo(self, current_output: Optional[np.ndarray]) -> Optional[PipelineHistoryEntry]:
        """Restore the previous state from the undo stack."""

        if not self._undo_stack:
            LOGGER.debug("Undo requested but stack empty")
            return None
        self._redo_stack.append(self._entry_for_current_state(current_output))
        entry = self._undo_stack.pop()
        self.steps = [step.clone() for step in entry.steps]
        LOGGER.info("Undo applied (remaining undo depth=%s)", len(self._undo_stack))
        restored = entry.clone()
        self._last_execution_entry = restored.clone()
        return restored

    def redo(self, current_output: Optional[np.ndarray]) -> Optional[PipelineHistoryEntry]:
        """Restore the next state from the redo stack."""

        if not self._redo_stack:
            LOGGER.debug("Redo requested but stack empty")
            return None
        self._undo_stack.append(self._entry_for_current_state(current_output))
        entry = self._redo_stack.pop()
        self.steps = [step.clone() for step in entry.steps]
        LOGGER.info("Redo applied (remaining redo depth=%s)", len(self._redo_stack))
        restored = entry.clone()
        self._last_execution_entry = restored.clone()
        return restored

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def history_depth(self) -> tuple[int, int]:
        """Return the number of undo and redo states currently cached."""

        return (len(self._undo_stack), len(self._redo_stack))

    # ------------------------------------------------------------------
    # Execution and serialisation helpers
    # ------------------------------------------------------------------
    def apply(
        self,
        image: PipelineImage,
        *,
        cache_threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES,
        cache_dir: Optional[os.PathLike[str] | str] = None,
    ) -> PipelineImage:
        """Execute the pipeline against ``image`` returning the processed copy.

        While executing, intermediate step outputs are cached so that undo/redo
        flows can restore previous results without recomputing earlier steps.
        ``cache_threshold_bytes`` controls when outputs spill to disk backed
        storage. ``cache_dir`` can be used to redirect disk backed artefacts.
        """

        if isinstance(image, TiledPipelineImage) and self._should_stream_tiled():
            return self._apply_streaming_pipeline(
                image,
                cache_threshold_bytes=cache_threshold_bytes,
                cache_dir=cache_dir,
            )
        return self._apply_eager_pipeline(
            image,
            cache_threshold_bytes=cache_threshold_bytes,
            cache_dir=cache_dir,
        )

    def _apply_eager_pipeline(
        self,
        image: PipelineImage,
        *,
        cache_threshold_bytes: int,
        cache_dir: Optional[os.PathLike[str] | str],
    ) -> PipelineImage:
        processed: PipelineImage = image.copy() if isinstance(image, NDArray) else image
        intermediates: Dict[str, CachedArray] = {}
        self._last_failure = None
        cache_directory = os.fspath(cache_dir) if cache_dir is not None else self._cache_dir
        for step in self.steps:
            try:
                processed = self._run_step(step, processed)
            except Exception as exc:
                traceback_text = traceback.format_exc()
                recovery_path = self._write_recovery_trace(step.name, traceback_text)
                step.enabled = False
                failure = PipelineFailure(step.name, exc, traceback_text, recovery_path)
                self._last_failure = failure
                LOGGER.error(
                    "Pipeline step failed",
                    exc_info=exc,
                    extra={
                        "step": step.name,
                        "recovery_path": str(recovery_path),
                    },
                )
                raise PipelineExecutionError(failure) from exc

            cache_ready = processed if isinstance(processed, NDArray) else processed.to_array()
            intermediates[step.name] = CachedArray.from_array(
                cache_ready,
                threshold_bytes=cache_threshold_bytes,
                cache_dir=cache_directory,
            )
        final_array = processed if isinstance(processed, NDArray) else processed.to_array()
        final_cache = CachedArray.from_optional(
            final_array,
            threshold_bytes=cache_threshold_bytes,
            cache_dir=cache_directory,
        )
        entry = PipelineHistoryEntry(
            self._snapshot_steps(),
            final_cache,
            intermediates,
        )
        self._last_execution_entry = entry.clone()
        return processed

    def _should_stream_tiled(self) -> bool:
        for step in self.steps:
            if step.enabled and not step.supports_tiled_input:
                return True
        return False

    def _apply_streaming_pipeline(
        self,
        image: TiledPipelineImage,
        *,
        cache_threshold_bytes: int,
        cache_dir: Optional[os.PathLike[str] | str],
    ) -> np.ndarray:
        intermediates: Dict[str, CachedArray] = {}
        self._last_failure = None
        cache_directory = os.fspath(cache_dir) if cache_dir is not None else self._cache_dir
        tile_size = image.tile_size
        shape = image.infer_shape()
        current_array: Optional[np.ndarray] = None
        tile_boxes: Optional[List[TileBox]] = None

        for step in self.steps:
            try:
                current_array, tile_boxes = self._stream_step(
                    step,
                    image,
                    current_array,
                    tile_boxes,
                    shape,
                    tile_size,
                )
            except Exception as exc:
                traceback_text = traceback.format_exc()
                recovery_path = self._write_recovery_trace(step.name, traceback_text)
                step.enabled = False
                failure = PipelineFailure(step.name, exc, traceback_text, recovery_path)
                self._last_failure = failure
                LOGGER.error(
                    "Pipeline step failed",
                    exc_info=exc,
                    extra={
                        "step": step.name,
                        "recovery_path": str(recovery_path),
                    },
                )
                raise PipelineExecutionError(failure) from exc

            intermediates[step.name] = CachedArray.from_array(
                current_array,
                threshold_bytes=cache_threshold_bytes,
                cache_dir=cache_directory,
            )

        if current_array is None:
            current_array = image.to_array()

        final_cache = CachedArray.from_optional(
            current_array,
            threshold_bytes=cache_threshold_bytes,
            cache_dir=cache_directory,
        )
        entry = PipelineHistoryEntry(
            self._snapshot_steps(),
            final_cache,
            intermediates,
        )
        self._last_execution_entry = entry.clone()
        return current_array

    def _stream_step(
        self,
        step: PipelineStep,
        image: TiledPipelineImage,
        current_array: Optional[np.ndarray],
        tile_boxes: Optional[List[TileBox]],
        shape: Tuple[int, ...],
        tile_size: Optional[TileSize],
    ) -> Tuple[np.ndarray, List[TileBox]]:
        boxes = tile_boxes if tile_boxes is not None else []
        output: Optional[np.ndarray] = None

        if current_array is None:
            iterator = image.iter_tiles(tile_size)
        else:
            if not boxes:
                raise ValueError("tile_boxes must be defined when streaming subsequent steps")
            iterator = ((box, self._extract_tile(current_array, box)) for box in boxes)

        for index, (box, tile) in enumerate(iterator):
            if tile_boxes is None:
                boxes.append(box)
            operand: PipelineImage = np.array(tile, copy=True)
            result = self._run_step(step, operand)
            if isinstance(result, TiledPipelineImage):
                result = result.to_array()
            tile_array = np.array(result, copy=False)
            if output is None:
                output = np.zeros(shape, dtype=tile_array.dtype)
            self._paste_tile(output, box, tile_array)

        if output is None:
            if current_array is not None:
                output = np.array(current_array, copy=True)
            else:
                output = image.to_array()
                if not boxes:
                    height, width = output.shape[0], output.shape[1]
                    boxes = [(0, 0, width, height)]

        return output, boxes

    @staticmethod
    def _extract_tile(array: np.ndarray, box: TileBox) -> np.ndarray:
        left, top, right, bottom = box
        slices = (slice(top, bottom), slice(left, right))
        if array.ndim > 2:
            slices += (slice(None),)
        return array[slices]

    @staticmethod
    def _paste_tile(target: np.ndarray, box: TileBox, tile: np.ndarray) -> None:
        left, top, right, bottom = box
        if target.ndim == 2:
            target[top:bottom, left:right] = tile
        else:
            target[top:bottom, left:right, ...] = tile

    def _run_step(self, step: PipelineStep, image: PipelineImage) -> PipelineImage:
        """Execute ``step`` honouring execution metadata."""

        if not step.enabled:
            return image

        if step.execution.requires_gpu:
            array_input = image if isinstance(image, NDArray) else image.to_array()
            if self._gpu_executor is None:
                LOGGER.warning(
                    "Step '%s' requires GPU execution but no executor is configured; falling back to CPU.",
                    step.name,
                )
                return step.apply(array_input)
            result = self._gpu_executor.execute(step, array_input)
            if result is None:
                result = array_input
            if step.execution.supports_inplace and isinstance(result, NDArray):
                if result is array_input:
                    return array_input
                if result.shape == array_input.shape and result.dtype == array_input.dtype:
                    array_input[...] = result
                    return array_input
            return result

        return step.apply(image)

    def last_failure(self) -> Optional[PipelineFailure]:
        """Return the most recent pipeline failure, if any."""

        return self._last_failure

    def _write_recovery_trace(self, step_name: str, traceback_text: str) -> Path:
        timestamp = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in step_name)
        recovery_dir = self._recovery_root / f"{timestamp}_{safe_name}_{uuid.uuid4().hex[:8]}"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        trace_path = recovery_dir / "traceback.txt"
        trace_path.write_text(traceback_text, encoding="utf-8")
        return trace_path

    def get_cached_output(self, step_name: str) -> Optional[np.ndarray]:
        """Return cached data for ``step_name`` from the latest application."""

        if self._last_execution_entry is None:
            return None
        try:
            return self._last_execution_entry.get_cached_output(step_name)
        except KeyError:
            return None

    def replay_from_cache(self) -> List[tuple[str, np.ndarray]]:
        """Return cached step outputs for the most recent pipeline run."""

        if self._last_execution_entry is None:
            return []
        return list(self._last_execution_entry.replay_from_cache())

    def get_order(self) -> List[str]:
        return [step.name for step in self.steps]

    def to_dict(self, include_cache: bool = False) -> Dict[str, Any]:
        """Serialise the pipeline configuration.

        Parameters
        ----------
        include_cache:
            When ``True``, include cached outputs from the most recent run in
            the serialized payload.
        """

        data: Dict[str, Any] = {"steps": [step.to_dict() for step in self.steps]}
        if include_cache and self._last_execution_entry is not None:
            data["cache"] = self._last_execution_entry.to_dict()
        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        function_registry: Dict[str, Callable[..., np.ndarray]] | Callable[[str], Callable[..., np.ndarray]],
    ) -> "PipelineManager":
        """Recreate a manager from :meth:`to_dict` output.

        ``function_registry`` can either be a mapping or a callable. When a
        step cannot be resolved a warning is logged and the step is skipped.
        """

        if callable(function_registry):
            resolver = function_registry
        else:
            resolver = function_registry.__getitem__

        steps: List[PipelineStep] = []
        for item in data.get("steps", []):
            try:
                steps.append(PipelineStep.from_dict(item, resolver))
            except KeyError:
                LOGGER.warning("Unknown pipeline step '%s' skipped during load", item.get("name"))
        manager = cls(steps)
        cache_blob = data.get("cache")
        if cache_blob:
            try:
                manager._last_execution_entry = PipelineHistoryEntry.from_dict(cache_blob, resolver)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Failed to restore pipeline cache: %s", exc)
        return manager

    # ------------------------------------------------------------------
    # Introspection helpers used by UIs
    # ------------------------------------------------------------------
    def iter_steps(self) -> Iterable[PipelineStep]:
        return iter(self.steps)

    def __len__(self) -> int:  # pragma: no cover - container convenience
        return len(self.steps)

    def __iter__(self):  # pragma: no cover - container convenience
        return iter(self.steps)

@dataclass
class PipelineFailure:
    """Report generated when a pipeline step fails during execution."""

    step_name: str
    exception: Exception
    traceback: str
    recovery_path: Path


class PipelineExecutionError(RuntimeError):
    """Error raised when a pipeline step cannot be executed successfully."""

    def __init__(self, failure: PipelineFailure) -> None:
        message = f"Pipeline step '{failure.step_name}' failed: {failure.exception}"
        super().__init__(message)
        self.failure = failure

