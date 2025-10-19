"""Unified application shell hosting multiple processing stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore
from plugins.module_base import ModuleStage
from processing.pipeline_manager import PipelineImage, PipelineManager, PipelineStep
from ui import ModulePane
from ui.theme import ThemedDockWidget
from yam_processor.ui.diagnostics_panel import DiagnosticsPanel


StageToolbar = Tuple[QtWidgets.QToolBar, QtCore.Qt.ToolBarArea]
StageDock = Tuple[QtWidgets.QDockWidget, QtCore.Qt.DockWidgetArea]


@dataclass
class StageRegistration:
    """Hold the metadata associated with a registered stage pane."""

    pane: ModulePane
    toolbars: List[StageToolbar]
    docks: List[StageDock]
    status_message: str | None


class UnifiedMainWindow(QtWidgets.QMainWindow):
    """High level window coordinating stage panes, toolbars and diagnostics."""

    def __init__(
        self,
        app_core: AppCore,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.app_core = app_core
        self.setObjectName("unifiedMainWindow")
        canonical_title = self.tr("Yam Image Processor")
        self.setWindowTitle(canonical_title)
        self._canonical_title = canonical_title
        self._canonical_icon = QtGui.QIcon(self.windowIcon())
        self.setProperty("isUnifiedShell", True)
        self.resize(1400, 900)

        self._tab_widget = QtWidgets.QTabWidget(self)
        self._tab_widget.setObjectName("stageTabWidget")
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        container = QtWidgets.QWidget(self)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self._tab_widget)
        self.setCentralWidget(container)

        self._stage_order: List[ModuleStage] = []
        self._registered_stages: set[ModuleStage] = set()
        self._active_stage: ModuleStage | None = None
        self._stage_registry: Dict[ModuleStage, StageRegistration] = {}

        self._stage_label = QtWidgets.QLabel(self)
        self._stage_label.setObjectName("activeStageLabel")
        self.statusBar().addPermanentWidget(self._stage_label, 0)
        self.statusBar().showMessage(self.tr("Ready"))
        self._update_stage_indicator(None)

        self._diagnostics_panel = DiagnosticsPanel(self)
        self._diagnostics_dock = ThemedDockWidget(self.tr("Diagnostics Log"), self)
        self._diagnostics_dock.setObjectName("diagnosticsDock")
        self._diagnostics_dock.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea
            | QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
        )
        self._diagnostics_dock.setWidget(self._diagnostics_panel)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._diagnostics_dock)
        self._diagnostics_dock.visibilityChanged.connect(
            self._on_diagnostics_visibility_changed
        )

        handler = self._diagnostics_panel.log_handler()
        if self.app_core.log_handler is not None and getattr(
            self.app_core.log_handler, "formatter", None
        ) is not None:
            handler.setFormatter(self.app_core.log_handler.formatter)
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
        self._diagnostics_panel.attach_to_logger(logging.getLogger())
        if self.app_core.thread_controller is not None:
            self._diagnostics_panel.set_thread_controller(self.app_core.thread_controller)

    def is_unified_shell(self) -> bool:
        """Return ``True`` to advertise unified shell hosting capabilities."""

        return True

    # ------------------------------------------------------------------
    # Stage registration helpers
    # ------------------------------------------------------------------
    def add_stage_pane(
        self,
        stage: ModuleStage,
        widget: QtWidgets.QWidget,
        title: str,
        *,
        toolbars: Sequence[StageToolbar] | None = None,
        docks: Sequence[StageDock] | None = None,
        status_message: str | None = None,
    ) -> None:
        """Register ``widget`` as the pane responsible for ``stage``."""

        if stage in self._registered_stages:
            raise ValueError(f"Stage {stage!r} already registered")

        pane = self._coerce_module_pane(widget)

        index = self._tab_widget.addTab(pane, title)
        pane.setObjectName(f"stagePane_{stage.name.lower()}")
        self._stage_order.insert(index, stage)
        prepared_toolbars: List[StageToolbar] = []
        for toolbar, area in toolbars or ():
            toolbar.setParent(self)
            prepared_toolbars.append((toolbar, area))
        prepared_docks: List[StageDock] = []
        for dock, area in docks or ():
            dock.setParent(self)
            prepared_docks.append((dock, area))

        self._registered_stages.add(stage)
        self._stage_registry[stage] = StageRegistration(
            pane=pane,
            toolbars=prepared_toolbars,
            docks=prepared_docks,
            status_message=status_message,
        )

        if self._tab_widget.count() == 1:
            self._tab_widget.setCurrentIndex(0)
            self._activate_stage(stage)

    def add_stage_toolbars(
        self,
        stage: ModuleStage,
        toolbars: Iterable[StageToolbar],
    ) -> None:
        """Append ``toolbars`` to ``stage`` after initial registration."""

        registration = self._get_registration(stage)
        for toolbar, area in toolbars:
            toolbar.setParent(self)
            registration.toolbars.append((toolbar, area))
        if stage == self._active_stage:
            for toolbar, area in toolbars:
                self.addToolBar(area, toolbar)
                toolbar.show()

    def add_stage_docks(
        self,
        stage: ModuleStage,
        docks: Iterable[StageDock],
    ) -> None:
        """Append ``docks`` to ``stage`` after initial registration."""

        registration = self._get_registration(stage)
        for dock, area in docks:
            dock.setParent(self)
            registration.docks.append((dock, area))
        if stage == self._active_stage:
            for dock, area in docks:
                self.addDockWidget(area, dock)
                dock.show()

    def set_stage_status(self, stage: ModuleStage, message: str | None) -> None:
        """Update the persistent status message displayed for ``stage``."""

        registration = self._get_registration(stage)
        registration.status_message = message
        if stage == self._active_stage:
            self._apply_stage_status(stage)

    def load_image(self) -> None:
        """Delegate image loading to the active module pane."""

        pane = self._active_pane()
        if pane is None:
            logging.getLogger(__name__).debug("No active pane to load an image")
            return
        pane.load_image()

    def save_outputs(self) -> None:
        """Delegate output saving to the active module pane."""

        pane = self._active_pane()
        if pane is None:
            logging.getLogger(__name__).debug("No active pane to save outputs")
            return
        pane.save_outputs()

    def update_pipeline_summary(self) -> None:
        """Delegate pipeline summary updates to the active module pane."""

        pane = self._active_pane()
        if pane is None:
            logging.getLogger(__name__).debug(
                "No active pane to update pipeline summary"
            )
            return
        pane.update_pipeline_summary()

    def toggle_diagnostics_dock(self, visible: bool) -> None:
        """Toggle diagnostics visibility and inform the active pane."""

        self._diagnostics_dock.setVisible(visible)
        self._notify_diagnostics_visibility(visible)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _on_tab_changed(self, index: int) -> None:
        if not (0 <= index < len(self._stage_order)):
            return
        stage = self._stage_order[index]
        self._activate_stage(stage)

    def _activate_stage(self, stage: ModuleStage) -> None:
        if stage == self._active_stage:
            return

        if self._active_stage is not None:
            previous_registration = self._get_registration(self._active_stage)
            for toolbar, _ in previous_registration.toolbars:
                self.removeToolBar(toolbar)
                toolbar.hide()
            for dock, _ in previous_registration.docks:
                self.removeDockWidget(dock)
                dock.hide()
            try:
                previous_registration.pane.on_deactivated()
            except Exception:  # pragma: no cover - defensive cleanup
                logging.getLogger(__name__).exception(
                    "Error while deactivating stage %s", self._active_stage
                )

        registration = self._get_registration(stage)
        for toolbar, area in registration.toolbars:
            self.addToolBar(area, toolbar)
            toolbar.show()
        for dock, area in registration.docks:
            self.addDockWidget(area, dock)
            dock.show()

        self._active_stage = stage
        self._update_stage_indicator(stage)
        self._apply_stage_status(stage)
        try:
            registration.pane.refresh_menus()
        except Exception:  # pragma: no cover - defensive menu refresh
            logging.getLogger(__name__).exception(
                "Error while refreshing menus for stage %s", stage
            )
        try:
            registration.pane.on_activated()
        except Exception:  # pragma: no cover - defensive activation
            logging.getLogger(__name__).exception(
                "Error while activating stage %s", stage
            )
        self._notify_diagnostics_visibility(self._diagnostics_dock.isVisible())
        self._restore_window_chrome()

    def _apply_stage_status(self, stage: ModuleStage) -> None:
        registration = self._get_registration(stage)
        message = registration.status_message
        if message:
            self.statusBar().showMessage(message)
        else:
            self.statusBar().showMessage(self.tr("Ready"))

    def _update_stage_indicator(self, stage: ModuleStage | None) -> None:
        if stage is None:
            self._stage_label.setText(self.tr("No stage selected"))
            return
        title = self._tab_widget.tabText(self._tab_widget.currentIndex())
        self._stage_label.setText(self.tr("Stage: {title}").format(title=title))

    def _active_pane(self) -> ModulePane | None:
        if self._active_stage is None:
            return None
        return self._get_registration(self._active_stage).pane

    def _get_registration(self, stage: ModuleStage) -> StageRegistration:
        try:
            return self._stage_registry[stage]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Stage {stage!r} is not registered") from exc

    def _on_diagnostics_visibility_changed(self, visible: bool) -> None:
        self._notify_diagnostics_visibility(visible)

    def _notify_diagnostics_visibility(self, visible: bool) -> None:
        pane = self._active_pane()
        if pane is None:
            return
        try:
            pane.set_diagnostics_visible(visible)
        except Exception:  # pragma: no cover - defensive notification
            logging.getLogger(__name__).exception(
                "Error updating diagnostics visibility for %s", pane.objectName()
            )

    def _restore_window_chrome(self) -> None:
        """Reinstate the canonical title and icon for the unified shell."""

        if self.windowTitle() != self._canonical_title:
            self.setWindowTitle(self._canonical_title)
        if self.windowIcon().cacheKey() != self._canonical_icon.cacheKey():
            self.setWindowIcon(self._canonical_icon)

    def _coerce_module_pane(self, widget: QtWidgets.QWidget) -> ModulePane:
        if not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("Stage panes must be QWidget instances")

        required_methods = (
            "on_activated",
            "on_deactivated",
            "load_image",
            "save_outputs",
            "update_pipeline_summary",
            "set_diagnostics_visible",
            "teardown",
        )
        missing = [
            name for name in required_methods if not callable(getattr(widget, name, None))
        ]
        if missing:
            raise TypeError(
                "Stage panes must implement the ModulePane interface; missing "
                + ", ".join(sorted(missing))
            )

        return cast(ModulePane, widget)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        try:
            self._diagnostics_panel.detach_from_logger()
        except Exception:  # pragma: no cover - defensive
            pass
        for registration in self._stage_registry.values():
            try:
                registration.pane.teardown()
            except Exception:  # pragma: no cover - defensive cleanup
                logging.getLogger(__name__).exception(
                    "Error during teardown of pane %s", registration.pane.objectName()
                )
        super().closeEvent(event)


class UnifiedPipelineController(QtCore.QObject):
    """Coordinate unified pipeline operations across module stages."""

    stage_steps_inserted = QtCore.pyqtSignal(object, int, object)
    stage_steps_removed = QtCore.pyqtSignal(object, int, object)
    stage_steps_updated = QtCore.pyqtSignal(object, int, object)
    stage_cache_updated = QtCore.pyqtSignal(object, object)
    pipeline_recomputed = QtCore.pyqtSignal(object)

    def __init__(
        self, app_core: AppCore, parent: Optional[QtCore.QObject] = None
    ) -> None:
        super().__init__(parent)
        self._app_core = app_core
        self._pipeline_manager: PipelineManager = app_core.get_pipeline_manager()
        try:
            self._pipeline_manager.add_change_listener(self._on_pipeline_manager_changed)
        except AttributeError:  # pragma: no cover - defensive fallback when listener missing
            LOGGER.debug("Pipeline manager does not support change listeners", exc_info=True)
        self._stage_order: Tuple[ModuleStage, ...] = tuple(ModuleStage)
        self._stage_ranges: Dict[ModuleStage, Tuple[int, int]] = {}
        self._stage_cache: Dict[ModuleStage, Tuple[PipelineStep, ...]] = {}
        self._stage_results: Dict[ModuleStage, PipelineImage] = {}
        self._stage_dependencies: Dict[ModuleStage, Tuple[ModuleStage, ...]] = (
            self._build_stage_dependencies()
        )
        self._combined_cache: Tuple[PipelineStep, ...] = ()
        self.recompute_pipeline()

    # ------------------------------------------------------------------
    # Stage information helpers
    # ------------------------------------------------------------------
    def pipeline_manager(self) -> PipelineManager:
        """Return the shared :class:`PipelineManager` instance."""

        return self._pipeline_manager

    def pipeline_stage_bounds(self, stage: ModuleStage) -> Tuple[int, int]:
        """Return the current (start, end) bounds for ``stage``."""

        return self._stage_ranges[stage]

    def stage_slice(self, stage: ModuleStage) -> slice:
        """Return a slice targeting ``stage`` within the unified pipeline."""

        start, end = self.pipeline_stage_bounds(stage)
        return slice(start, end)

    def stage_steps(self, stage: ModuleStage) -> Tuple[PipelineStep, ...]:
        """Return the live pipeline steps backing ``stage``."""

        start, end = self.pipeline_stage_bounds(stage)
        return tuple(self._pipeline_manager.steps[start:end])

    def cached_stage_steps(self, stage: ModuleStage) -> Tuple[PipelineStep, ...]:
        """Return a cached snapshot of the steps owned by ``stage``."""

        return self._stage_cache.get(stage, ())

    def cached_pipeline(self) -> Tuple[PipelineStep, ...]:
        """Return a cached snapshot of the entire pipeline."""

        return self._combined_cache

    def stage_dependencies(self, stage: ModuleStage) -> Tuple[ModuleStage, ...]:
        """Return the upstream stage dependencies for ``stage``."""

        return self._stage_dependencies.get(stage, ())

    def cached_stage_result(self, stage: ModuleStage) -> Optional[PipelineImage]:
        """Return the most recent cached image produced by ``stage``."""

        return self._stage_results.get(stage)

    def cached_stage_results(self) -> Dict[ModuleStage, PipelineImage]:
        """Return a shallow copy of all cached stage results."""

        return dict(self._stage_results)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def insert_stage_step(
        self,
        stage: ModuleStage,
        step: PipelineStep,
        index: Optional[int] = None,
    ) -> None:
        """Insert ``step`` into ``stage`` at ``index`` (append if ``None``)."""

        insert_at, global_index = self._resolve_insert_index(stage, index)
        normalised = self._normalise_step(stage, step)
        self._pipeline_manager.add_step(normalised, index=global_index)
        self.stage_steps_inserted.emit(stage, insert_at, normalised.clone())

    def remove_stage_step(self, stage: ModuleStage, index: int) -> PipelineStep:
        """Remove and return the step at ``index`` within ``stage``."""

        global_index = self._resolve_index(stage, index)
        removed = self._pipeline_manager.remove_step(global_index)
        self.stage_steps_removed.emit(stage, index, removed.clone())
        return removed

    def mutate_stage_step(
        self,
        stage: ModuleStage,
        index: int,
        mutator: Callable[[PipelineStep], None],
    ) -> PipelineStep:
        """Apply ``mutator`` to the step at ``index`` within ``stage``."""

        global_index = self._resolve_index(stage, index)
        step = self._pipeline_manager.get_step(global_index)
        mutator(step)
        self._refresh_caches(affected_stages=(stage,))
        self.stage_steps_updated.emit(stage, index, step.clone())
        return step

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def recompute_pipeline(
        self, *, preserve_stage_results: bool = False
    ) -> Tuple[PipelineStep, ...]:
        """Recompute cached snapshots for every stage and the full pipeline."""

        steps = tuple(self._pipeline_manager.steps)
        self._recalculate_stage_ranges(steps)
        self._stage_dependencies = self._build_stage_dependencies()
        if not preserve_stage_results:
            self._stage_results.clear()
        self._combined_cache = tuple(step.clone() for step in steps)
        for stage in self._stage_order:
            start, end = self._stage_ranges[stage]
            stage_steps = steps[start:end]
            self._stage_cache[stage] = tuple(step.clone() for step in stage_steps)
            self.stage_cache_updated.emit(stage, self._stage_cache[stage])
        self.pipeline_recomputed.emit(self._combined_cache)
        return self._combined_cache

    def run_enabled_stages(
        self,
        source_image: PipelineImage,
        *,
        seeded_results: Optional[Mapping[ModuleStage, PipelineImage]] = None,
    ) -> Dict[ModuleStage, PipelineImage]:
        """Execute enabled stages sequentially, caching intermediate images.

        Parameters
        ----------
        source_image:
            The initial image supplied to the first stage in the pipeline.
        seeded_results:
            Optional mapping providing precomputed outputs for specific stages.
            When supplied, seeded stages will not be re-executed and downstream
            stages will consume the provided results instead.

        Returns
        -------
        Dict[ModuleStage, PipelineImage]
            A mapping of stage identifiers to the most recent image produced
            after that stage has executed.
        """

        stage_results: Dict[ModuleStage, PipelineImage] = {}
        if seeded_results:
            stage_results.update(seeded_results)
        for stage in self._stage_order:
            dependencies = self._stage_dependencies.get(stage, ())
            stage_input: PipelineImage = source_image
            for dependency in dependencies:
                if dependency in stage_results:
                    stage_input = stage_results[dependency]

            if stage in stage_results:
                result = stage_results[stage]
            else:
                result = stage_input
                executed = False
                for step in self.stage_steps(stage):
                    if not step.enabled:
                        continue
                    executed = True
                    result = step.apply(result)

                if not executed:
                    result = stage_input

                stage_results[stage] = result

        self._stage_results = dict(stage_results)
        return dict(stage_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_index(self, stage: ModuleStage, index: int) -> int:
        start, end = self.pipeline_stage_bounds(stage)
        length = end - start
        if index < 0 or index >= length:
            raise IndexError(f"Stage index {index} out of range for {stage}")
        return start + index

    def _resolve_insert_index(
        self, stage: ModuleStage, index: Optional[int]
    ) -> Tuple[int, int]:
        start, end = self.pipeline_stage_bounds(stage)
        length = end - start
        insert_at = length if index is None else index
        if insert_at < 0 or insert_at > length:
            raise IndexError(f"Insert index {insert_at} out of range for {stage}")
        return insert_at, start + insert_at

    def _normalise_step(self, stage: ModuleStage, step: PipelineStep) -> PipelineStep:
        if getattr(step, "stage", None) != stage:
            clone = step.clone()
            clone.stage = stage
            return clone
        return step

    def invalidate_stage_results(self, *stages: ModuleStage) -> None:
        """Invalidate cached results for ``stages`` and their dependants."""

        self._invalidate_stage_results(stages)

    def _refresh_caches(
        self, *, affected_stages: Optional[Iterable[ModuleStage]] = None
    ) -> None:
        if affected_stages is None:
            targets = self._stage_order
        else:
            stage_set = {stage for stage in affected_stages if stage is not None}
            targets = tuple(stage for stage in self._stage_order if stage in stage_set)
            if not targets:
                targets = self._stage_order
        self.recompute_pipeline(preserve_stage_results=True)
        self._invalidate_stage_results(targets)

    def _recalculate_stage_ranges(self, steps: Sequence[PipelineStep]) -> None:
        index = 0
        total = len(steps)
        for stage in self._stage_order:
            start = index
            while index < total and getattr(steps[index], "stage", None) == stage:
                index += 1
            self._stage_ranges[stage] = (start, index)

    def _build_stage_dependencies(self) -> Dict[ModuleStage, Tuple[ModuleStage, ...]]:
        dependencies: Dict[ModuleStage, Tuple[ModuleStage, ...]] = {}
        upstream: List[ModuleStage] = []
        for stage in self._stage_order:
            dependencies[stage] = tuple(upstream)
            upstream.append(stage)
        return dependencies

    def _expand_downstream(
        self, stages: Iterable[ModuleStage]
    ) -> Tuple[ModuleStage, ...]:
        stage_set = {stage for stage in stages if stage is not None}
        if not stage_set:
            return ()
        ordered: List[ModuleStage] = []
        for stage in self._stage_order:
            if stage in stage_set:
                ordered.append(stage)
                continue
            dependencies = self._stage_dependencies.get(stage, ())
            if any(dep in stage_set for dep in dependencies):
                ordered.append(stage)
        return tuple(ordered)

    def _invalidate_stage_results(self, stages: Iterable[ModuleStage]) -> None:
        invalidated = self._expand_downstream(stages)
        if not invalidated:
            return
        for stage in invalidated:
            self._stage_results.pop(stage, None)

    def _stages_from_payload(self, payload: Mapping[str, Any]) -> Tuple[ModuleStage, ...]:
        stages: List[ModuleStage] = []
        single = payload.get("step")
        if isinstance(single, PipelineStep):
            stage = getattr(single, "stage", None)
            if stage is not None:
                stages.append(stage)
        steps_payload = payload.get("steps")
        if isinstance(steps_payload, (list, tuple)):
            for entry in steps_payload:
                if not isinstance(entry, PipelineStep):
                    continue
                stage = getattr(entry, "stage", None)
                if stage is None or stage in stages:
                    continue
                stages.append(stage)
        return tuple(stages)

    def _on_pipeline_manager_changed(self, event: str, payload: Dict[str, Any]) -> None:
        affected = self._stages_from_payload(payload)
        if not affected or event in {"pipeline_reset", "steps_replaced", "pipeline_restored"}:
            affected = self._stage_order
        self._refresh_caches(affected_stages=affected)


__all__ = [
    "UnifiedMainWindow",
    "UnifiedPipelineController",
    "StageToolbar",
    "StageDock",
    "StageRegistration",
]

