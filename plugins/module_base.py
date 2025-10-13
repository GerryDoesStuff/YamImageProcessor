"""Base classes for dynamically discovered processing modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Tuple

import numpy as np

from processing.pipeline_manager import PipelineStep

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from ui.preprocessing import MainWindow


class ModuleStage(Enum):
    """Enumerate the supported stages of the processing pipeline."""

    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    ANALYSIS = "analysis"


@dataclass(frozen=True)
class ModuleMetadata:
    """Describes a module for menu presentation and diagnostics."""

    identifier: str
    title: str
    stage: ModuleStage
    description: str = ""
    menu_path: Tuple[str, ...] = ("Pre-Processing",)
    shortcut: str | None = None
    default_enabled: bool = False


@dataclass(frozen=True)
class MenuEntry:
    """Represents a menu entry exposed by a module."""

    path: Tuple[str, ...]
    text: str
    description: str = ""
    shortcut: str | None = None


class ModuleBase(ABC):
    """Base class for processing modules discovered at runtime."""

    def __init__(self) -> None:
        self._metadata = self._build_metadata()

    # ------------------------------------------------------------------
    # Metadata helpers
    @property
    def metadata(self) -> ModuleMetadata:
        """Return the descriptive metadata for this module."""

        return self._metadata

    @abstractmethod
    def _build_metadata(self) -> ModuleMetadata:
        """Construct the immutable metadata descriptor for the module."""

    # ------------------------------------------------------------------
    # Menu registration helpers
    def menu_entries(self) -> Sequence[MenuEntry]:
        """Return the menu entries contributed by the module."""

        metadata = self.metadata
        return (
            MenuEntry(
                path=metadata.menu_path,
                text=metadata.title,
                description=metadata.description,
                shortcut=metadata.shortcut,
            ),
        )

    def activate(self, window: "MainWindow") -> None:
        """Execute the module's UI interaction when invoked from a menu."""

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement an activation handler"
        )

    # ------------------------------------------------------------------
    # Pipeline registration helpers
    def default_parameters(self) -> dict[str, Any]:
        """Return the default parameters for the module's pipeline step."""

        return {}

    def create_pipeline_step(self) -> PipelineStep:
        """Create a :class:`PipelineStep` template for this module."""

        return PipelineStep(
            name=self.metadata.identifier,
            function=self.process,
            enabled=self.metadata.default_enabled,
            params=self.default_parameters(),
        )

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Execute the module's processing logic."""

    # ------------------------------------------------------------------
    # Bulk registration helpers
    @classmethod
    def iter_modules(cls) -> Iterable[type["ModuleBase"]]:  # pragma: no cover - helper
        """Yield all subclasses of :class:`ModuleBase` defined on the class."""

        for subclass in cls.__subclasses__():
            yield subclass


__all__ = [
    "MenuEntry",
    "ModuleBase",
    "ModuleMetadata",
    "ModuleStage",
]
