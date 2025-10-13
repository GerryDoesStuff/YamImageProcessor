"""Pipeline utilities for the preprocessing tool."""
from __future__ import annotations

from typing import Iterable, Optional

from core.app_core import AppCore
from processing.pipeline_manager import PipelineManager, PipelineStep


class PreprocessingPipeline(PipelineManager):
    """Runtime pipeline used when executing the preprocessing steps."""

    def __init__(
        self,
        app_core: Optional[AppCore] = None,
        steps: Optional[Iterable[PipelineStep]] = None,
    ) -> None:
        super().__init__(steps)
        self.app_core = app_core
        self.thread_controller = getattr(app_core, "thread_controller", None)


def build_preprocessing_pipeline(
    app_core: AppCore,
    manager: Optional[PipelineManager] = None,
) -> PreprocessingPipeline:
    """Construct a :class:`PreprocessingPipeline` from ``manager``."""

    if manager is None:
        manager = app_core.get_preprocessing_pipeline_manager()
    pipeline = PreprocessingPipeline(app_core, manager.steps)
    return pipeline


__all__ = [
    "PreprocessingPipeline",
    "PipelineManager",
    "PipelineStep",
    "build_preprocessing_pipeline",
]
