"""Controller utilities for working with :class:`PipelineManager`."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from PyQt5 import QtWidgets  # type: ignore

from yam_processor.processing import PipelineManager, PipelineStep

from .dialogs import ParameterDialog, ParameterSpec


LOGGER = logging.getLogger(__name__)


class PipelineController:
    """Coordinates rebuild, toggling, and history actions for a pipeline."""

    def __init__(self, builder: Callable[[], PipelineManager]) -> None:
        self._builder = builder
        self.manager: PipelineManager = builder()
        self._open_dialogs: list[ParameterDialog] = []
        LOGGER.debug("PipelineController initialised with steps: %s", self.manager.get_order())

    def rebuild_pipeline(self) -> PipelineManager:
        """Recreate the pipeline using the stored builder."""

        self.manager = self._builder()
        LOGGER.debug("Pipeline rebuilt with steps: %s", self.manager.get_order())
        return self.manager

    # ------------------------------------------------------------------
    # Parameter dialog helpers
    # ------------------------------------------------------------------
    def create_parameter_dialog(
        self,
        identifier: int | str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        source_image: Optional[np.ndarray] = None,
    ) -> ParameterDialog:
        """Construct a modeless :class:`ParameterDialog` for ``identifier``."""

        step = self.manager.get_step(identifier)
        schema = self._resolve_schema(step)
        preview_callable = self._resolve_preview_callable(step)

        dialog = ParameterDialog(
            schema=schema,
            preview_callback=preview_callable,
            parent=parent,
            window_title=f"{step.name} Parameters",
        )
        dialog.set_parameters(step.params)
        if source_image is not None:
            dialog.set_source_image(source_image)

        initial_params = step.params.copy()
        dialog.parametersChanged.connect(lambda params, step=step: self._update_step_params(step, params))
        dialog.cancelled.connect(lambda step=step, params=initial_params: self._on_dialog_cancelled(step, params))
        dialog.finished.connect(lambda _result, dlg=dialog: self._untrack_dialog(dlg))
        dialog.destroyed.connect(lambda _obj=None, dlg=dialog: self._untrack_dialog(dlg))
        self._open_dialogs.append(dialog)
        return dialog

    def open_parameter_dialog(
        self,
        identifier: int | str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        source_image: Optional[np.ndarray] = None,
    ) -> ParameterDialog:
        """Create and show a parameter dialog for ``identifier``."""

        dialog = self.create_parameter_dialog(identifier, parent=parent, source_image=source_image)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_schema(self, step: PipelineStep) -> Sequence[ParameterSpec]:
        function = step.function
        schema_provider = getattr(function, "parameter_schema", None)
        if callable(schema_provider):
            return list(schema_provider())

        module = getattr(function, "__self__", None)
        if module is not None:
            provider = getattr(module, "parameter_schema", None)
            if callable(provider):
                return list(provider())
        return []

    def _resolve_preview_callable(
        self, step: PipelineStep
    ) -> Callable[[np.ndarray, Dict[str, Any]], np.ndarray]:
        function = step.function

        preview_provider = getattr(function, "preview", None)
        if callable(preview_provider):
            return lambda image, params, provider=preview_provider: provider(image, **params)

        module = getattr(function, "__self__", None)
        if module is not None:
            provider = getattr(module, "preview", None)
            if callable(provider):
                return lambda image, params, provider=provider: provider(image, **params)

        return lambda image, params, function=function: function(image, **params)

    def _update_step_params(self, step: PipelineStep, params: Dict[str, Any]) -> None:
        step.params.clear()
        step.params.update(params)

    def _on_dialog_cancelled(self, step: PipelineStep, params: Dict[str, Any]) -> None:
        self._update_step_params(step, params)

    def _untrack_dialog(self, dialog: ParameterDialog) -> None:
        try:
            self._open_dialogs.remove(dialog)
        except ValueError:
            pass

