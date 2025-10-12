"""High level Qt window composition for Yam Image Processor."""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

from .pipeline_controller import PipelineController


class MainWindow(QtWidgets.QMainWindow):
    """Primary application window with dockable panels and status feedback."""

    pipelineDockVisibilityChanged = QtCore.pyqtSignal(bool)
    previewDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    statusMessageRequested = QtCore.pyqtSignal(str, int)
    openProjectRequested = QtCore.pyqtSignal()
    saveProjectRequested = QtCore.pyqtSignal()
    saveProjectAsRequested = QtCore.pyqtSignal()
    exitRequested = QtCore.pyqtSignal()
    undoRequested = QtCore.pyqtSignal(object)
    redoRequested = QtCore.pyqtSignal(object)
    manageModulesRequested = QtCore.pyqtSignal()
    documentationRequested = QtCore.pyqtSignal()
    aboutRequested = QtCore.pyqtSignal()

    def __init__(
        self,
        pipeline_controller: Optional[PipelineController] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        self._pipeline_controller: Optional[PipelineController] = None
        if pipeline_controller is not None:
            self.set_pipeline_controller(pipeline_controller)

        self.setWindowTitle("Yam Image Processor")

        self._central_widget = QtWidgets.QWidget(self)
        self._central_layout = QtWidgets.QVBoxLayout(self._central_widget)
        self._central_layout.setObjectName("centralLayout")
        self._apply_scaled_layout_metrics()
        self._central_layout.addStretch(1)
        self.setCentralWidget(self._central_widget)

        self._build_actions()
        self._build_menus()
        self._build_docks()
        self._build_status_bar()

        self.screenChanged.connect(self._on_screen_changed)
        self.statusMessageRequested.connect(self.show_status_message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pipeline_controller(self, controller: PipelineController) -> None:
        """Bind the window to a :class:`PipelineController`."""

        self._pipeline_controller = controller

    @QtCore.pyqtSlot(bool)
    def toggle_pipeline_dock(self, visible: bool) -> None:
        self.pipeline_dock.setVisible(visible)

    @QtCore.pyqtSlot(bool)
    def toggle_preview_dock(self, visible: bool) -> None:
        self.preview_dock.setVisible(visible)

    @QtCore.pyqtSlot(bool)
    def toggle_diagnostics_dock(self, visible: bool) -> None:
        self.diagnostics_dock.setVisible(visible)

    @QtCore.pyqtSlot(str, int)
    def show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        self.statusBar().showMessage(message, timeout_ms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_status_bar(self) -> None:
        status_bar = QtWidgets.QStatusBar(self)
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

    def _build_actions(self) -> None:
        self.open_project_action = QtWidgets.QAction("&Open Project…", self)
        self.open_project_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_project_action.setStatusTip("Open a pipeline configuration from disk")
        self.open_project_action.triggered.connect(self._on_open_project)

        self.save_project_action = QtWidgets.QAction("&Save Project", self)
        self.save_project_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_project_action.setStatusTip("Persist the active pipeline configuration")
        self.save_project_action.triggered.connect(self._on_save_project)

        self.save_project_as_action = QtWidgets.QAction("Save Project &As…", self)
        self.save_project_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_project_as_action.setStatusTip("Persist the active pipeline configuration to a new file")
        self.save_project_as_action.triggered.connect(self._on_save_project_as)

        self.exit_action = QtWidgets.QAction("E&xit", self)
        self.exit_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Quit))
        self.exit_action.setStatusTip("Close Yam Image Processor")
        self.exit_action.triggered.connect(self._on_exit_requested)

        self.undo_action = QtWidgets.QAction("&Undo", self)
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)
        self.undo_action.setStatusTip("Undo the previous pipeline change")
        self.undo_action.triggered.connect(self._on_undo_requested)

        self.redo_action = QtWidgets.QAction("&Redo", self)
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)
        self.redo_action.setStatusTip("Redo the previously undone pipeline change")
        self.redo_action.triggered.connect(self._on_redo_requested)

        self.manage_modules_action = QtWidgets.QAction("&Manage Modules…", self)
        self.manage_modules_action.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self.manage_modules_action.setStatusTip("Open the module manager")
        self.manage_modules_action.triggered.connect(self.manageModulesRequested.emit)

        self.documentation_action = QtWidgets.QAction("&Documentation", self)
        self.documentation_action.setShortcut(QtGui.QKeySequence.HelpContents)
        self.documentation_action.setStatusTip("Open the user documentation")
        self.documentation_action.triggered.connect(self.documentationRequested.emit)

        self.about_action = QtWidgets.QAction("&About", self)
        self.about_action.setStatusTip("Show information about Yam Image Processor")
        self.about_action.triggered.connect(self.aboutRequested.emit)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu("&File")
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = menu_bar.addMenu("&Edit")
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)

        self.view_menu = menu_bar.addMenu("&View")

        self.modules_menu = menu_bar.addMenu("&Modules")
        self.modules_menu.addAction(self.manage_modules_action)

        self.help_menu = menu_bar.addMenu("&Help")
        self.help_menu.addAction(self.documentation_action)
        self.help_menu.addAction(self.about_action)

    def _build_docks(self) -> None:
        self.pipeline_dock = self._create_dock(
            "Pipeline", "Pipeline configuration and ordering controls"
        )
        self.preview_dock = self._create_dock(
            "Preview", "Pipeline output preview"
        )
        self.diagnostics_dock = self._create_dock(
            "Diagnostics", "Log output and performance metrics"
        )

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.pipeline_dock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.preview_dock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.diagnostics_dock)

        self.pipeline_dock.visibilityChanged.connect(
            self.pipelineDockVisibilityChanged.emit
        )
        self.preview_dock.visibilityChanged.connect(
            self.previewDockVisibilityChanged.emit
        )
        self.diagnostics_dock.visibilityChanged.connect(
            self.diagnosticsDockVisibilityChanged.emit
        )

        self.pipeline_toggle_action = self.pipeline_dock.toggleViewAction()
        self.preview_toggle_action = self.preview_dock.toggleViewAction()
        self.diagnostics_toggle_action = self.diagnostics_dock.toggleViewAction()

        self.view_menu.addAction(self.pipeline_toggle_action)
        self.view_menu.addAction(self.preview_toggle_action)
        self.view_menu.addAction(self.diagnostics_toggle_action)

        self._install_context_menus()

    def _create_dock(self, title: str, placeholder_text: str) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(f"{title.lower()}Dock")
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
            | QtCore.Qt.BottomDockWidgetArea
        )
        dock_content = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(dock_content)
        layout.setContentsMargins(*self._scaled_margins(12))
        layout.setSpacing(self._scaled_value(6))
        label = QtWidgets.QLabel(placeholder_text, dock_content)
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        dock.setWidget(dock_content)
        return dock

    def _install_context_menus(self) -> None:
        pipeline_widget = self.pipeline_dock.widget()
        self._assign_actions_to_widget(
            pipeline_widget,
            (
                self.open_project_action,
                self.save_project_action,
                self.save_project_as_action,
                None,
                self.undo_action,
                self.redo_action,
            ),
        )

        preview_widget = self.preview_dock.widget()
        self._assign_actions_to_widget(
            preview_widget,
            (
                self.pipeline_toggle_action,
                self.preview_toggle_action,
                self.diagnostics_toggle_action,
            ),
        )

        diagnostics_widget = self.diagnostics_dock.widget()
        self._assign_actions_to_widget(
            diagnostics_widget,
            (
                self.pipeline_toggle_action,
                self.preview_toggle_action,
                self.diagnostics_toggle_action,
            ),
        )

    def _assign_actions_to_widget(
        self, widget: Optional[QtWidgets.QWidget], actions: Iterable[Optional[QtWidgets.QAction]]
    ) -> None:
        if widget is None:
            return
        widget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        for action in actions:
            if action is None:
                separator = QtWidgets.QAction(widget)
                separator.setSeparator(True)
                widget.addAction(separator)
            else:
                widget.addAction(action)

    def _on_screen_changed(self, screen: Optional[QtGui.QScreen]) -> None:
        self._apply_scaled_layout_metrics()
        for dock in (self.pipeline_dock, self.preview_dock, self.diagnostics_dock):
            content = dock.widget()
            if content is None:
                continue
            layout = content.layout()
            if isinstance(layout, QtWidgets.QVBoxLayout):
                layout.setContentsMargins(*self._scaled_margins(12, screen))
                layout.setSpacing(self._scaled_value(6, screen))

    def _apply_scaled_layout_metrics(self) -> None:
        margins = self._scaled_margins(16)
        spacing = self._scaled_value(10)
        self._central_layout.setContentsMargins(*margins)
        self._central_layout.setSpacing(spacing)

    def _scaled_value(self, base: int, screen: Optional[QtGui.QScreen] = None) -> int:
        screen = screen or self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return base
        scale = max(screen.logicalDotsPerInch() / 96.0, 1.0)
        return max(1, round(base * scale))

    def _scaled_margins(
        self, base: int, screen: Optional[QtGui.QScreen] = None
    ) -> tuple[int, int, int, int]:
        value = self._scaled_value(base, screen)
        return (value, value, value, value)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    @QtCore.pyqtSlot()
    def _on_open_project(self) -> None:
        self.openProjectRequested.emit()

    @QtCore.pyqtSlot()
    def _on_save_project(self) -> None:
        self.saveProjectRequested.emit()

    @QtCore.pyqtSlot()
    def _on_save_project_as(self) -> None:
        self.saveProjectAsRequested.emit()

    @QtCore.pyqtSlot()
    def _on_exit_requested(self) -> None:
        self.exitRequested.emit()
        self.close()

    @QtCore.pyqtSlot()
    def _on_undo_requested(self) -> None:
        result = None
        if self._pipeline_controller is not None:
            result = self._pipeline_controller.undo(None)
        self.undoRequested.emit(result)

    @QtCore.pyqtSlot()
    def _on_redo_requested(self) -> None:
        result = None
        if self._pipeline_controller is not None:
            result = self._pipeline_controller.redo(None)
        self.redoRequested.emit(result)

__all__ = ["MainWindow"]

