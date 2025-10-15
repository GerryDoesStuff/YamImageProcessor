"""Controller utilities for working with :class:`PipelineManager`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import numpy as np

from PyQt5 import QtWidgets  # type: ignore

from yam_processor.data import ImageRecord, TiledImageRecord, load_image, save_image
from yam_processor.processing import PipelineManager, PipelineStep

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from yam_processor.core.persistence import AutosaveManager

from .dialogs import ParameterDialog, ParameterSpec


LOGGER = logging.getLogger(__name__)


class PipelineController:
    """Coordinates rebuild, toggling, and history actions for a pipeline."""

    def __init__(
        self,
        builder: Callable[[], PipelineManager],
        *,
        autosave_manager: Optional["AutosaveManager"] = None,
    ) -> None:
        self._builder = builder
        self.manager: PipelineManager = builder()
        self._open_dialogs: list[ParameterDialog] = []
        self._autosave_manager: Optional["AutosaveManager"] = autosave_manager
        LOGGER.debug("PipelineController initialised with steps: %s", self.manager.get_order())

    def rebuild_pipeline(self) -> PipelineManager:
        """Recreate the pipeline using the stored builder."""

        self.manager = self._builder()
        LOGGER.debug("Pipeline rebuilt with steps: %s", self.manager.get_order())
        self._mark_pipeline_dirty({"action": "rebuild"})
        return self.manager

    # ------------------------------------------------------------------
    # Image I/O helpers
    # ------------------------------------------------------------------
    def load_image_record(
        self, path: str | Path, *, record_history: bool = False
    ) -> ImageRecord | TiledImageRecord:
        """Load an image file into an :class:`ImageRecord`.

        Parameters
        ----------
        path:
            Location of the image or ``.npy`` file on disk.
        record_history:
            When ``True`` the loaded pixel data is appended to the pipeline
            history so it can participate in undo/redo operations.
        """

        record = load_image(path)
        if record_history:
            self.record_history(record.to_array())
        return record

    def save_image_record(
        self,
        record: ImageRecord | TiledImageRecord,
        path: str | Path,
        *,
        format: Optional[str] = None,
    ) -> None:
        """Persist ``record`` using the shared image I/O helpers."""

        save_image(record, path, format)

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
        error_metadata = {"step": step.name}
        module = getattr(step.function, "__self__", None)
        if module is not None:
            error_metadata["module"] = module.__class__.__name__
            module_metadata = getattr(module, "metadata", None)
            module_name = getattr(module_metadata, "name", None)
            if module_name:
                error_metadata["moduleName"] = module_name

        dialog = ParameterDialog(
            schema=schema,
            preview_callback=preview_callable,
            parent=parent,
            window_title=f"{step.name} Parameters",
            error_metadata=error_metadata,
        )
        dialog.set_parameters(step.params)
        if source_image is not None:
            dialog.set_source_image(source_image)

        initial_params = step.params.copy()
        dialog.parametersChanged.connect(lambda params, step=step: self._update_step_params(step, params))
        dialog.cancelled.connect(lambda step=step, params=initial_params: self._on_dialog_cancelled(step, params))
        dialog.finished.connect(lambda _result, dlg=dialog: self._untrack_dialog(dlg))
        dialog.destroyed.connect(lambda _obj=None, dlg=dialog: self._untrack_dialog(dlg))
        dialog.errorOccurred.connect(self._on_dialog_error)
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
        self._mark_pipeline_dirty({"action": "enable", "identifier": identifier})

    def disable_step(self, identifier: int | str) -> None:
        self.manager.set_step_enabled(identifier, False)
        self._mark_pipeline_dirty({"action": "disable", "identifier": identifier})

    def toggle_step(self, identifier: int | str) -> None:
        self.manager.toggle_step(identifier)
        self._mark_pipeline_dirty({"action": "toggle", "identifier": identifier})

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
        self._mark_pipeline_dirty({"action": "undo"})
        return entry.output

    def redo(self, current_output: Optional[np.ndarray]) -> Optional[np.ndarray]:
        entry = self.manager.redo(current_output)
        if entry is None:
            LOGGER.debug("Redo requested but no state available")
            return None
        LOGGER.info("Redo restored pipeline order: %s", [step.name for step in entry.steps])
        self._mark_pipeline_dirty({"action": "redo"})
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
        self._mark_pipeline_dirty({"action": "update_params", "step": step.name})

    def _on_dialog_cancelled(self, step: PipelineStep, params: Dict[str, Any]) -> None:
        self._update_step_params(step, params)

    def _untrack_dialog(self, dialog: ParameterDialog) -> None:
        try:
            self._open_dialogs.remove(dialog)
        except ValueError:
            pass

    def _on_dialog_error(self, operation: str, payload: dict) -> None:
        context = dict(payload)
        context.setdefault("operation", operation)
        LOGGER.error("Parameter dialog error", extra={"dialog_error": context})

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def set_autosave_manager(self, manager: Optional["AutosaveManager"]) -> None:
        self._autosave_manager = manager

    def save_project(
        self, destination: Optional[Path | str], *, metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        if self._autosave_manager is None:
            raise RuntimeError("Autosave manager has not been configured")

        payload = self.manager.to_dict()
        combined_metadata = self._build_metadata(metadata)
        return self._autosave_manager.save(destination, pipeline=payload, metadata=combined_metadata)

    def _mark_pipeline_dirty(self, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._autosave_manager is None:
            return
        payload = self.manager.to_dict()
        metadata = self._build_metadata(extra_metadata)
        try:
            self._autosave_manager.mark_dirty(payload, metadata)
        except Exception as exc:  # pragma: no cover - autosave failures should not crash UI
            LOGGER.error("Failed to schedule autosave", exc_info=exc)

    def _build_metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "stepCount": len(self.manager.steps),
            "steps": self.manager.get_order(),
        }
        if extra:
            metadata.update(extra)
        return metadata

