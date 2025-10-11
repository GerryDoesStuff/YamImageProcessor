"""Controller utilities for working with :class:`PipelineManager`."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from yam_processor.processing import PipelineManager


LOGGER = logging.getLogger(__name__)


class PipelineController:
    """Coordinates rebuild, toggling, and history actions for a pipeline."""

    def __init__(self, builder: Callable[[], PipelineManager]) -> None:
        self._builder = builder
        self.manager: PipelineManager = builder()
        LOGGER.debug("PipelineController initialised with steps: %s", self.manager.get_order())

    def rebuild_pipeline(self) -> PipelineManager:
        """Recreate the pipeline using the stored builder."""

        self.manager = self._builder()
        LOGGER.debug("Pipeline rebuilt with steps: %s", self.manager.get_order())
        return self.manager

    # ------------------------------------------------------------------
    # Step toggling helpers
    # ------------------------------------------------------------------
    def enable_step(self, identifier: int | str) -> None:
        self.manager.set_step_enabled(identifier, True)

    def disable_step(self, identifier: int | str) -> None:
        self.manager.set_step_enabled(identifier, False)

    def toggle_step(self, identifier: int | str) -> None:
        self.manager.toggle_step(identifier)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def record_history(self, output: Optional[np.ndarray]) -> None:
        self.manager.push_history(output)
        LOGGER.info("History recorded (undo=%s, redo=%s)", self.manager.can_undo(), self.manager.can_redo())

    def undo(self, current_output: Optional[np.ndarray]) -> Optional[np.ndarray]:
        entry = self.manager.undo(current_output)
        if entry is None:
            LOGGER.debug("Undo requested but no state available")
            return None
        LOGGER.info("Undo restored pipeline order: %s", [step.name for step in entry.steps])
        return entry.output

    def redo(self, current_output: Optional[np.ndarray]) -> Optional[np.ndarray]:
        entry = self.manager.redo(current_output)
        if entry is None:
            LOGGER.debug("Redo requested but no state available")
            return None
        LOGGER.info("Redo restored pipeline order: %s", [step.name for step in entry.steps])
        return entry.output

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return self.manager.to_dict()

