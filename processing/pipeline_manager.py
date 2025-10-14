"""Utilities for managing ordered image processing pipelines."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class PipelineStep:
    """A single step in an image processing pipeline."""

    name: str
    function: Callable[..., np.ndarray]
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Execute the processing step if enabled."""

        if not self.enabled:
            return image
        return self.function(image, **self.params)

    def clone(self) -> "PipelineStep":
        """Return a deep copy of the step."""

        return PipelineStep(
            name=self.name,
            function=self.function,
            enabled=self.enabled,
            params=dict(self.params),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "params": dict(self.params),
        }


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
    ) -> None:
        template = [step.clone() for step in steps or []]
        self._template: List[PipelineStep] = template
        self._steps: List[PipelineStep] = [step.clone() for step in template]
        self._undo_stack: List[PipelineState] = []
        self._redo_stack: List[PipelineState] = []
        self._cache_directory: Optional[Path] = None
        self._recovery_root: Optional[Path] = None
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

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        for step in self.iter_enabled_steps():
            result = step.apply(result)
        return result

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
    "PipelineManager",
    "PipelineState",
    "PipelineStep",
]

