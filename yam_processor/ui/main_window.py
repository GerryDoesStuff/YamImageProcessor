"""High level Qt window composition for Yam Image Processor."""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

from .pipeline_controller import PipelineController


class MainWindow(QtWidgets.QMainWindow):
    """Primary application window with dockable panels and status feedback."""

    pipelineDockVisibilityChanged = QtCore.pyqtSignal(bool)
    previewDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    statusMessageRequested = QtCore.pyqtSignal(str, int)

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

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.pipeline_dock.toggleViewAction())
        view_menu.addAction(self.preview_dock.toggleViewAction())
        view_menu.addAction(self.diagnostics_dock.toggleViewAction())

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


__all__ = ["MainWindow"]

