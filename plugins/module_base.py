"""Base classes for dynamically discovered processing modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, Tuple

import numpy as np

from processing.pipeline_manager import PipelineStep, StepExecutionMetadata
from processing.tiled_records import TiledPipelineImage

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from ui.preprocessing import MainWindow
    from ui.control_metadata import ControlMetadata


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

    def _load_parameter_metadata(self) -> Mapping[str, "ControlMetadata"]:
        try:
            from ui.control_metadata import get_module_control_metadata
        except Exception:  # pragma: no cover - defensive fallback
            return {}
        return get_module_control_metadata(self.metadata.identifier)

    def parameter_metadata(self) -> Mapping[str, "ControlMetadata"]:
        """Return the registered control metadata for this module, if any."""

        return self._load_parameter_metadata()

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

        defaults: dict[str, Any] = {}
        for name, metadata in self.parameter_metadata().items():
            if metadata.default is not None:
                defaults[name] = metadata.default
        return defaults

    def sanitize_parameters(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Normalise ``params`` using the metadata registry where available."""

        sanitised: dict[str, Any] = dict(self.default_parameters())
        sanitised.update(params)
        for name, metadata in self.parameter_metadata().items():
            if name in sanitised:
                sanitised[name] = metadata.coerce(sanitised[name])
        return sanitised

    def pipeline_execution_metadata(self) -> StepExecutionMetadata:
        """Return execution hints for the module's pipeline step."""

        return StepExecutionMetadata()

    def supports_tiled_input(self) -> bool:
        """Return ``True`` if the module can process tiled image handles."""

        return False

    def create_pipeline_step(self) -> PipelineStep:
        """Create a :class:`PipelineStep` template for this module."""

        return PipelineStep(
            name=self.metadata.identifier,
            function=self.process,
            enabled=self.metadata.default_enabled,
            params=self.default_parameters(),
            execution=self.pipeline_execution_metadata(),
            supports_tiled_input=self.supports_tiled_input(),
        )

    @abstractmethod
    def process(
        self, image: np.ndarray | TiledPipelineImage, **kwargs: Any
    ) -> np.ndarray | TiledPipelineImage:
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
