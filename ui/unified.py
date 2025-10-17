"""Unified application shell hosting multiple processing stages."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore
from plugins.module_base import ModuleStage
from ui.theme import ThemedDockWidget
from yam_processor.ui.diagnostics_panel import DiagnosticsPanel


StageToolbar = Tuple[QtWidgets.QToolBar, QtCore.Qt.ToolBarArea]
StageDock = Tuple[QtWidgets.QDockWidget, QtCore.Qt.DockWidgetArea]


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
        self.setWindowTitle(self.tr("Yam Image Processor"))
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
        self._toolbar_cache: Dict[ModuleStage, List[StageToolbar]] = {}
        self._dock_cache: Dict[ModuleStage, List[StageDock]] = {}
        self._stage_status: Dict[ModuleStage, str | None] = {}

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

        index = self._tab_widget.addTab(widget, title)
        widget.setObjectName(f"stagePane_{stage.name.lower()}")
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
        self._toolbar_cache[stage] = prepared_toolbars
        self._dock_cache[stage] = prepared_docks
        self._stage_status[stage] = status_message

        if self._tab_widget.count() == 1:
            self._tab_widget.setCurrentIndex(0)
            self._activate_stage(stage)

    def add_stage_toolbars(
        self,
        stage: ModuleStage,
        toolbars: Iterable[StageToolbar],
    ) -> None:
        """Append ``toolbars`` to ``stage`` after initial registration."""

        self._toolbar_cache.setdefault(stage, [])
        for toolbar, area in toolbars:
            toolbar.setParent(self)
            self._toolbar_cache[stage].append((toolbar, area))
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

        self._dock_cache.setdefault(stage, [])
        for dock, area in docks:
            dock.setParent(self)
            self._dock_cache[stage].append((dock, area))
        if stage == self._active_stage:
            for dock, area in docks:
                self.addDockWidget(area, dock)
                dock.show()

    def set_stage_status(self, stage: ModuleStage, message: str | None) -> None:
        """Update the persistent status message displayed for ``stage``."""

        self._stage_status[stage] = message
        if stage == self._active_stage:
            self._apply_stage_status(stage)

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
            for toolbar, _ in self._toolbar_cache.get(self._active_stage, []):
                self.removeToolBar(toolbar)
                toolbar.hide()
            for dock, _ in self._dock_cache.get(self._active_stage, []):
                self.removeDockWidget(dock)
                dock.hide()

        for toolbar, area in self._toolbar_cache.get(stage, []):
            self.addToolBar(area, toolbar)
            toolbar.show()
        for dock, area in self._dock_cache.get(stage, []):
            self.addDockWidget(area, dock)
            dock.show()

        self._active_stage = stage
        self._update_stage_indicator(stage)
        self._apply_stage_status(stage)

    def _apply_stage_status(self, stage: ModuleStage) -> None:
        message = self._stage_status.get(stage)
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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        try:
            self._diagnostics_panel.detach_from_logger()
        except Exception:  # pragma: no cover - defensive
            pass
        super().closeEvent(event)


__all__ = ["UnifiedMainWindow", "StageToolbar", "StageDock"]
