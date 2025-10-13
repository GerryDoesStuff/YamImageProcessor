"""Base classes and helpers for plugin implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Sequence, TypeVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import guard for typing
    from yam_processor.core.app_core import AppCore
    from yam_processor.ui.dialogs import ParameterSpec


__all__ = [
    "ModuleMetadata",
    "ModuleCapabilities",
    "PipelineStage",
    "ModuleBase",
    "PreprocessingModule",
    "SegmentationModule",
    "AnalysisModule",
    "register_module",
]


@dataclass
class ModuleMetadata:
    """Describes a pipeline module for UI presentation and diagnostics."""

    name: str
    version: str = "0.1.0"
    description: str = ""


@dataclass
class ModuleCapabilities:
    """Capability flags advertised by a module implementation.

    Modules that mutate their input buffers in place or that require a GPU
    accelerator should mirror those requirements onto their pipeline steps via
    :class:`~yam_processor.processing.StepExecutionMetadata`. The metadata is
    serialised with the pipeline, allowing the runtime to reuse numpy buffers
    when ``supports_inplace`` is set and to delegate GPU bound work when
    ``requires_gpu`` is ``True``. Authors can construct the metadata directly::

        StepExecutionMetadata(supports_inplace=True, requires_gpu=True)

    and attach it when creating :class:`~yam_processor.processing.PipelineStep`
    instances. Keeping the module-level capability flags and per-step metadata
    aligned helps user interfaces communicate the module's execution
    characteristics accurately.
    """

    supports_batch: bool = False
    requires_gpu: bool = False
    is_deterministic: bool = True


class PipelineStage(Enum):
    """Enumerates the supported stages of the processing pipeline."""

    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    ANALYSIS = "analysis"


class ModuleBase(ABC):
    """Common interface for all pipeline modules."""

    stage: ClassVar[PipelineStage]

    @property
    @abstractmethod
    def metadata(self) -> ModuleMetadata:
        """Return the descriptive metadata for the module."""

    @property
    def capabilities(self) -> ModuleCapabilities:
        """Advertised capability flags for the module.

        Sub-classes may override to expose stage specific flags.
        """

        return ModuleCapabilities()

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs: Any) -> Any:
        """Execute the module logic against ``image``."""

    def parameter_schema(self) -> Sequence["ParameterSpec"]:
        """Return the parameter controls consumed by the module."""

        return ()

    def preview(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate a preview for the provided ``image``."""

        return np.array(image, copy=True)


class PreprocessingModule(ModuleBase, ABC):
    """Base class for modules that transform raw images before segmentation."""

    stage: ClassVar[PipelineStage] = PipelineStage.PREPROCESSING

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return a pre-processed copy of ``image``."""


class SegmentationModule(ModuleBase, ABC):
    """Base class for modules responsible for segmenting images."""

    stage: ClassVar[PipelineStage] = PipelineStage.SEGMENTATION

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return a mask or segmented variant of ``image``."""


class AnalysisModule(ModuleBase, ABC):
    """Base class for modules that analyse segmented imagery."""

    stage: ClassVar[PipelineStage] = PipelineStage.ANALYSIS

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Return structured analysis data derived from ``image``."""


ModuleType = TypeVar("ModuleType", bound=ModuleBase)


def register_module(app_core: "AppCore", *module_classes: type[ModuleType]) -> None:
    """Register ``module_classes`` with the application's module registry.

    Plugin packages are expected to expose a module-level ``register_module``
    function that calls this helper to advertise their implementations. Each
    provided class must inherit from :class:`ModuleBase` and declare the
    appropriate :class:`PipelineStage` via its inheritance chain.
    """

    if not module_classes:
        raise ValueError("At least one module class must be supplied for registration.")

    for module_class in module_classes:
        app_core.module_registry.register(module_class)
