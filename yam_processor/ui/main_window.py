"""High level Qt window composition for Yam Image Processor."""

from __future__ import annotations

import logging
import traceback
from typing import Iterable, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

from .error_dialog import ErrorDialog
from .diagnostics_panel import DiagnosticsPanel
from .pipeline_controller import PipelineController
from .resources import load_icon
from .tooltips import build_main_window_tooltips
from yam_processor.core.threading import ThreadController

if TYPE_CHECKING:
    from yam_processor.core.app_core import UpdateDispatcher, UpdateMetadata


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
    errorOccurred = QtCore.pyqtSignal(str, dict)

    def __init__(
        self,
        pipeline_controller: Optional[PipelineController] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        thread_controller: Optional[ThreadController] = None,
        update_dispatcher: Optional["UpdateDispatcher"] = None,
    ) -> None:
        super().__init__(parent)

        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.AA_UseHighDpiPixmaps, True
        )
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        self._apply_accessibility_preferences()

        self._logger = logging.getLogger(__name__)
        self._app_version = self._resolve_app_version()
        self._thread_controller: Optional[ThreadController] = thread_controller
        self._update_dispatcher: Optional["UpdateDispatcher"] = None
        self._pending_update: Optional["UpdateMetadata"] = None
        self._pipeline_controller: Optional[PipelineController] = None
        if pipeline_controller is not None:
            self.set_pipeline_controller(pipeline_controller)
        if update_dispatcher is not None:
            self.set_update_dispatcher(update_dispatcher)

        self.setWindowTitle(self.tr("Yam Image Processor"))

        self.diagnostics_panel: Optional[DiagnosticsPanel] = None

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
        self._configure_keyboard_navigation()

        self.screenChanged.connect(self._on_screen_changed)
        self.statusMessageRequested.connect(self.show_status_message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pipeline_controller(self, controller: PipelineController) -> None:
        """Bind the window to a :class:`PipelineController`."""

        self._pipeline_controller = controller
        self._apply_action_tooltips()

    def set_thread_controller(self, controller: ThreadController) -> None:
        """Bind the diagnostics panel to ``controller``."""

        self._thread_controller = controller
        if self.diagnostics_panel is not None:
            self.diagnostics_panel.set_thread_controller(controller)

    def set_update_dispatcher(self, dispatcher: "UpdateDispatcher") -> None:
        """Listen for update notifications emitted by ``dispatcher``."""

        if self._update_dispatcher is dispatcher:
            return
        if self._update_dispatcher is not None:
            self._update_dispatcher.remove_listener(self._on_update_available)
        self._update_dispatcher = dispatcher
        dispatcher.add_listener(self._on_update_available)

    def diagnostics_log_handler(self) -> Optional[logging.Handler]:
        """Return the logging handler streaming messages to the diagnostics panel."""

        if self.diagnostics_panel is None:
            return None
        return self.diagnostics_panel.log_handler()

    def _on_update_available(self, metadata: "UpdateMetadata") -> None:
        """Handle update notifications emitted by :class:`AppCore`."""

        self._pending_update = metadata
        message = self.tr(
            "Update {version} is available"
        ).format(version=metadata.version)
        self.statusMessageRequested.emit(message, 0)
        self._logger.info(
            "Update available", extra={"component": "MainWindow", "version": metadata.version}
        )

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

    def show_about_dialog(self) -> None:
        """Display an about dialog including version metadata and opt-in notice."""

        version = self._app_version or "unknown"
        title = self.tr("About Yam Image Processor")
        body = self.tr(
            "<p><strong>Yam Image Processor</strong></p>"
            "<p>Version: {version}</p>"
            "<p>Telemetry is disabled unless explicitly enabled in settings.</p>"
        ).format(version=version)
        QtWidgets.QMessageBox.about(self, title, body)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_app_version(self) -> str:
        try:
            from yam_processor import get_version
        except Exception:  # pragma: no cover - fallback for circular imports
            return "unknown"
        try:
            return get_version()
        except Exception:  # pragma: no cover - metadata access safety
            return "unknown"

    def _build_status_bar(self) -> None:
        status_bar = QtWidgets.QStatusBar(self)
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

    def acknowledge_available_update(self) -> None:
        """Acknowledge any pending update notification."""

        if self._pending_update is None or self._update_dispatcher is None:
            return
        acknowledged = self._pending_update
        self._pending_update = None
        self._update_dispatcher.acknowledge()
        message = self.tr(
            "Update {version} acknowledged"
        ).format(version=acknowledged.version)
        self.statusMessageRequested.emit(message, 5000)

    def _build_actions(self) -> None:
        self.open_project_action = QtWidgets.QAction(self.tr("&Open Project…"), self)
        self.open_project_action.setIcon(load_icon("open_project"))
        self.open_project_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_project_action.setStatusTip(
            self.tr("Open a pipeline configuration from disk")
        )
        self.open_project_action.triggered.connect(self._on_open_project)

        self.save_project_action = QtWidgets.QAction(self.tr("&Save Project"), self)
        self.save_project_action.setIcon(load_icon("save_project"))
        self.save_project_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_project_action.setStatusTip(
            self.tr("Persist the active pipeline configuration")
        )
        self.save_project_action.triggered.connect(self._on_save_project)

        self.save_project_as_action = QtWidgets.QAction(
            self.tr("Save Project &As…"), self
        )
        self.save_project_as_action.setIcon(load_icon("save_project_as"))
        self.save_project_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        self.save_project_as_action.setStatusTip(
            self.tr("Persist the active pipeline configuration to a new file")
        )
        self.save_project_as_action.triggered.connect(self._on_save_project_as)

        self.exit_action = QtWidgets.QAction(self.tr("E&xit"), self)
        self.exit_action.setIcon(load_icon("exit"))
        self.exit_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Quit))
        self.exit_action.setStatusTip(self.tr("Close Yam Image Processor"))
        self.exit_action.triggered.connect(self._on_exit_requested)

        self.undo_action = QtWidgets.QAction(self.tr("&Undo"), self)
        self.undo_action.setIcon(load_icon("undo"))
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)
        self.undo_action.setStatusTip(self.tr("Undo the previous pipeline change"))
        self.undo_action.triggered.connect(self._on_undo_requested)

        self.redo_action = QtWidgets.QAction(self.tr("&Redo"), self)
        self.redo_action.setIcon(load_icon("redo"))
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)
        self.redo_action.setStatusTip(
            self.tr("Redo the previously undone pipeline change")
        )
        self.redo_action.triggered.connect(self._on_redo_requested)

        self.manage_modules_action = QtWidgets.QAction(
            self.tr("&Manage Modules…"), self
        )
        self.manage_modules_action.setIcon(load_icon("manage_modules"))
        self.manage_modules_action.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self.manage_modules_action.setStatusTip(self.tr("Open the module manager"))
        self.manage_modules_action.triggered.connect(self.manageModulesRequested.emit)

        self.documentation_action = QtWidgets.QAction(
            self.tr("&Documentation"), self
        )
        self.documentation_action.setIcon(load_icon("documentation"))
        self.documentation_action.setShortcut(QtGui.QKeySequence.HelpContents)
        self.documentation_action.setStatusTip(self.tr("Open the user documentation"))
        self.documentation_action.triggered.connect(self.documentationRequested.emit)

        self.about_action = QtWidgets.QAction(self.tr("&About"), self)
        self.about_action.setIcon(load_icon("about"))
        about_status = self.tr("Show information about Yam Image Processor")
        if self._app_version and self._app_version != "unknown":
            about_status = self.tr(
                "Show information about Yam Image Processor (version {version})"
            ).format(version=self._app_version)
        self.about_action.setStatusTip(about_status)
        self.about_action.triggered.connect(self._on_about_triggered)

        self.focus_diagnostics_action = QtWidgets.QAction(
            self.tr("Focus Diagnostics"), self
        )
        self.focus_diagnostics_action.setStatusTip(
            self.tr("Show and focus the diagnostics panel")
        )
        self.focus_diagnostics_action.triggered.connect(self._on_focus_diagnostics)

        self.clear_diagnostics_action = QtWidgets.QAction(
            self.tr("Clear Diagnostics Log"), self
        )
        self.clear_diagnostics_action.setStatusTip(
            self.tr("Remove logged entries from the diagnostics panel")
        )
        self.clear_diagnostics_action.triggered.connect(self._on_clear_diagnostics)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu(self.tr("&File"))
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = menu_bar.addMenu(self.tr("&Edit"))
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)

        self.view_menu = menu_bar.addMenu(self.tr("&View"))

        self.modules_menu = menu_bar.addMenu(self.tr("&Modules"))
        self.modules_menu.addAction(self.manage_modules_action)

        self.help_menu = menu_bar.addMenu(self.tr("&Help"))
        self.help_menu.addAction(self.documentation_action)
        self.help_menu.addAction(self.about_action)

    def _build_docks(self) -> None:
        self.pipeline_dock = self._create_dock(
            self.tr("Pipeline"),
            self.tr("Pipeline configuration and ordering controls"),
            object_name="pipelineDock",
        )
        self.preview_dock = self._create_dock(
            self.tr("Preview"),
            self.tr("Pipeline output preview"),
            object_name="previewDock",
        )
        self.diagnostics_panel = DiagnosticsPanel(self)
        self.diagnostics_panel.attach_to_logger(logging.getLogger())
        if self._thread_controller is not None:
            self.diagnostics_panel.set_thread_controller(self._thread_controller)
        self.diagnostics_dock = self._create_dock(
            self.tr("Diagnostics"),
            self.tr("Log output and performance metrics"),
            object_name="diagnosticsDock",
            content_widget=self.diagnostics_panel,
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
        self.diagnostics_toggle_action.triggered.connect(
            self._on_diagnostics_toggle_triggered
        )

        self.view_menu.addAction(self.pipeline_toggle_action)
        self.view_menu.addAction(self.preview_toggle_action)
        self.view_menu.addAction(self.diagnostics_toggle_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.focus_diagnostics_action)
        self.view_menu.addAction(self.clear_diagnostics_action)

        self._install_context_menus()
        self._apply_action_tooltips()

    def _create_dock(
        self,
        title: str,
        placeholder_text: str,
        *,
        object_name: str,
        content_widget: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
            | QtCore.Qt.BottomDockWidgetArea
        )
        dock.setFocusPolicy(QtCore.Qt.StrongFocus)
        dock_content = QtWidgets.QWidget(dock)
        dock_content.setFocusPolicy(QtCore.Qt.StrongFocus)
        layout = QtWidgets.QVBoxLayout(dock_content)
        layout.setContentsMargins(*self._scaled_margins(12))
        layout.setSpacing(self._scaled_value(6))
        if content_widget is None:
            label = QtWidgets.QLabel(placeholder_text, dock_content)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setFocusPolicy(QtCore.Qt.StrongFocus)
            layout.addWidget(label)
        else:
            content_widget.setParent(dock_content)
            layout.addWidget(content_widget)
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
                None,
                self.focus_diagnostics_action,
                self.clear_diagnostics_action,
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

    def _apply_action_tooltips(self) -> None:
        if not hasattr(self, "open_project_action"):
            return

        tooltips = build_main_window_tooltips(
            self._pipeline_controller, version=self._app_version
        )
        action_mapping = {
            "open_project": getattr(self, "open_project_action", None),
            "save_project": getattr(self, "save_project_action", None),
            "save_project_as": getattr(self, "save_project_as_action", None),
            "exit": getattr(self, "exit_action", None),
            "undo": getattr(self, "undo_action", None),
            "redo": getattr(self, "redo_action", None),
            "manage_modules": getattr(self, "manage_modules_action", None),
            "documentation": getattr(self, "documentation_action", None),
            "about": getattr(self, "about_action", None),
            "pipeline_toggle": getattr(self, "pipeline_toggle_action", None),
            "preview_toggle": getattr(self, "preview_toggle_action", None),
            "diagnostics_toggle": getattr(self, "diagnostics_toggle_action", None),
            "diagnostics_focus": getattr(self, "focus_diagnostics_action", None),
            "diagnostics_clear": getattr(self, "clear_diagnostics_action", None),
        }

        for key, action in action_mapping.items():
            if action is None:
                continue
            tooltip = tooltips.get(key)
            if tooltip:
                action.setToolTip(tooltip)
                action.setWhatsThis(tooltip)

        dock_mapping = {
            "pipeline_toggle": getattr(self, "pipeline_dock", None),
            "preview_toggle": getattr(self, "preview_dock", None),
            "diagnostics_toggle": getattr(self, "diagnostics_dock", None),
        }

        for key, dock in dock_mapping.items():
            if dock is None:
                continue
            tooltip = tooltips.get(key)
            if tooltip:
                dock.setToolTip(tooltip)

    def _on_screen_changed(self, screen: Optional[QtGui.QScreen]) -> None:
        self._apply_accessibility_preferences(screen)
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

    def _apply_accessibility_preferences(
        self, screen: Optional[QtGui.QScreen] = None
    ) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        fusion_style = QtWidgets.QStyleFactory.create("Fusion")
        if fusion_style is not None:
            app.setStyle(fusion_style)

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(18, 18, 18))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.black)
        palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 120, 212))
        palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
        app.setPalette(palette)

        screen = screen or self.screen() or app.primaryScreen()
        scale = 1.0
        if screen is not None:
            scale = max(screen.logicalDotsPerInch() / 96.0, 1.0)

        base_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
        point_size = base_font.pointSizeF()
        if point_size <= 0:
            point_size = 11.0
        else:
            point_size = max(point_size, 11.0)
        base_font.setPointSizeF(point_size * scale)

        app.setFont(base_font)
        self.setFont(base_font)

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

    def _configure_keyboard_navigation(self) -> None:
        self._focus_shortcuts: list[QtWidgets.QShortcut] = []

        self._central_widget.setFocusPolicy(QtCore.Qt.StrongFocus)

        focus_widgets: list[QtWidgets.QWidget] = [self._central_widget]
        for dock in (self.pipeline_dock, self.preview_dock, self.diagnostics_dock):
            target = dock.widget()
            if target is not None:
                focus_widgets.append(target)

        for first, second in zip(focus_widgets, focus_widgets[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

        shortcut_map: list[
            tuple[str, QtWidgets.QWidget | QtWidgets.QDockWidget]
        ] = [("Alt+0", self._central_widget)]
        if self.pipeline_dock.widget() is not None:
            shortcut_map.append(("Alt+1", self.pipeline_dock))
        if self.preview_dock.widget() is not None:
            shortcut_map.append(("Alt+2", self.preview_dock))
        if self.diagnostics_dock.widget() is not None:
            shortcut_map.append(("Alt+3", self.diagnostics_dock))

        for sequence, target in shortcut_map:
            self._focus_shortcuts.append(self._create_focus_shortcut(sequence, target))

        if self._central_widget.isEnabled():
            self._central_widget.setFocus(QtCore.Qt.OtherFocusReason)

    def _create_focus_shortcut(
        self, sequence: str, target: QtWidgets.QWidget | QtWidgets.QDockWidget
    ) -> QtWidgets.QShortcut:
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
        shortcut.activated.connect(lambda t=target: self._focus_target(t))
        return shortcut

    def _focus_target(
        self, target: QtWidgets.QWidget | QtWidgets.QDockWidget | None
    ) -> None:
        if target is None:
            return
        if isinstance(target, QtWidgets.QDockWidget):
            if not target.isVisible():
                target.show()
            target.raise_()
            widget = target.widget()
            focus_target: QtWidgets.QWidget = widget if widget is not None else target
            focus_target.setFocus(QtCore.Qt.ShortcutFocusReason)
        else:
            target.setFocus(QtCore.Qt.ShortcutFocusReason)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    @QtCore.pyqtSlot()
    def _on_open_project(self) -> None:
        self.openProjectRequested.emit()

    def _on_about_triggered(self) -> None:
        self.show_about_dialog()
        self.aboutRequested.emit()

    @QtCore.pyqtSlot()
    def _on_save_project(self) -> None:
        if self._pipeline_controller is not None:
            try:
                self._pipeline_controller.save_project(None)
            except ValueError:
                self._logger.debug("Save requested without a known destination; deferring to Save As")
            except RuntimeError:
                self._logger.debug("Save requested but autosave manager not configured")
            except Exception as exc:  # pragma: no cover - Qt exception handling
                self._report_error(
                    "save_project",
                    self.tr("Failed to save project"),
                    exc,
                )
                return
            else:
                self.statusMessageRequested.emit(self.tr("Project saved"), 3000)
                return

        self.saveProjectRequested.emit()

    @QtCore.pyqtSlot()
    def _on_save_project_as(self) -> None:
        self.saveProjectAsRequested.emit()

    @QtCore.pyqtSlot()
    def _on_exit_requested(self) -> None:
        self.exitRequested.emit()
        self.close()

    @QtCore.pyqtSlot()
    def _on_focus_diagnostics(self) -> None:
        self._focus_target(self.diagnostics_dock)
        if self.diagnostics_panel is not None:
            self.diagnostics_panel.focus_logs()

    @QtCore.pyqtSlot()
    def _on_clear_diagnostics(self) -> None:
        if self.diagnostics_panel is not None:
            self.diagnostics_panel.clear_logs()

    @QtCore.pyqtSlot(bool)
    def _on_diagnostics_toggle_triggered(self, visible: bool) -> None:
        if visible:
            self._on_focus_diagnostics()

    @QtCore.pyqtSlot()
    def _on_undo_requested(self) -> None:
        result = None
        if self._pipeline_controller is not None:
            try:
                result = self._pipeline_controller.undo(None)
            except Exception as exc:  # pragma: no cover - Qt exception handling
                self._report_error(
                    "undo",
                    self.tr("Failed to undo pipeline change"),
                    exc,
                )
                return
        self.undoRequested.emit(result)
        self._apply_action_tooltips()

    @QtCore.pyqtSlot()
    def _on_redo_requested(self) -> None:
        result = None
        if self._pipeline_controller is not None:
            try:
                result = self._pipeline_controller.redo(None)
            except Exception as exc:  # pragma: no cover - Qt exception handling
                self._report_error(
                    "redo",
                    self.tr("Failed to redo pipeline change"),
                    exc,
                )
                return
        self.redoRequested.emit(result)
        self._apply_action_tooltips()

    def _report_error(self, operation: str, message: str, exc: Exception) -> None:
        """Centralised handler for UI level exceptions."""

        traceback_text = traceback.format_exc()
        metadata = {
            "operation": operation,
            "window": self.objectName() or self.__class__.__name__,
        }
        if self._pipeline_controller is not None:
            metadata["controller"] = type(self._pipeline_controller).__name__
        payload = dict(metadata)
        payload.update(
            {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "traceback": traceback_text,
            }
        )
        self._logger.error(message, exc_info=exc, extra={"operation": operation})
        ErrorDialog.present(
            message,
            traceback_text,
            parent=self,
            metadata=metadata,
            window_title=self.tr("Operation failed"),
        )
        self.errorOccurred.emit(operation, payload)
        self.statusMessageRequested.emit(message, 5000)

__all__ = ["MainWindow"]

