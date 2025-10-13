"""Qt widgets for the preprocessing application."""
from __future__ import annotations

import json
import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore
from core.preprocessing import Config, Loader
from core.thread_controller import OperationCancelled, ThreadController
from plugins.module_base import ModuleBase, ModuleStage
from processing.pipeline_cache import PipelineCacheResult
from processing.pipeline_manager import PipelineManager
from processing.preprocessing_pipeline import (
    PreprocessingPipeline,
    build_preprocessing_pipeline,
)
from ui.theme import (
    SectionWidget,
    ShortcutRegistry,
    ShortcutSummaryWidget,
    ThemedDockWidget,
    load_icon,
    scale_font,
)


class ImageDisplayWidget(QtWidgets.QLabel):
    def __init__(self, use_rgb_format: bool = False):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self.use_rgb_format = use_rgb_format
        self.setMinimumSize(1, 1)

    def set_image(self, image: np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.use_rgb_format:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QtGui.QImage(image.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimage)
        self.update_pixmap()

    def update_pixmap(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled)
        else:
            self.setPixmap(QtGui.QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()


class BrightnessContrastDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(float, int)

    def __init__(
        self,
        alpha: float = 1.0,
        beta: int = 0,
        preview_callback: Optional[Callable[[float, int], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Brightness / Contrast")
        self.initial_alpha = alpha
        self.initial_beta = beta

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 3.0)
        self.alpha_spin.setValue(alpha)
        self.alpha_spin.setSingleStep(0.1)

        self.beta_spin = QtWidgets.QSpinBox()
        self.beta_spin.setRange(-100, 100)
        self.beta_spin.setValue(beta)

        form_layout.addRow("Contrast (alpha):", self.alpha_spin)
        form_layout.addRow("Brightness (beta):", self.beta_spin)

        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.alpha_spin.valueChanged.connect(self.on_value_changed)
        self.beta_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        alpha, beta = self.get_values()
        self.parametersChanged.emit(alpha, beta)

    def get_values(self) -> Tuple[float, int]:
        return self.alpha_spin.value(), self.beta_spin.value()

    def reset_to_initial(self):
        self.alpha_spin.setValue(self.initial_alpha)
        self.beta_spin.setValue(self.initial_beta)


class GammaDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        gamma: float = 1.0,
        preview_callback: Optional[Callable[[float], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Gamma Correction")
        self.initial_gamma = gamma

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(gamma)

        form_layout.addRow("Gamma:", self.gamma_spin)

        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.gamma_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        self.parametersChanged.emit(self.gamma_spin.value())

    def get_value(self) -> float:
        return self.gamma_spin.value()

    def reset_to_initial(self):
        self.gamma_spin.setValue(self.initial_gamma)


class NormalizeDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        alpha: int = 0,
        beta: int = 255,
        preview_callback: Optional[Callable[[int, int], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Intensity Normalization")
        self.initial_alpha = alpha
        self.initial_beta = beta

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.alpha_spin = QtWidgets.QSpinBox()
        self.alpha_spin.setRange(0, 255)
        self.alpha_spin.setValue(alpha)

        self.beta_spin = QtWidgets.QSpinBox()
        self.beta_spin.setRange(0, 255)
        self.beta_spin.setValue(beta)

        form_layout.addRow("Alpha:", self.alpha_spin)
        form_layout.addRow("Beta:", self.beta_spin)

        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.alpha_spin.valueChanged.connect(self.on_value_changed)
        self.beta_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        alpha, beta = self.get_values()
        self.parametersChanged.emit(alpha, beta)

    def get_values(self) -> Tuple[int, int]:
        return self.alpha_spin.value(), self.beta_spin.value()

    def reset_to_initial(self):
        self.alpha_spin.setValue(self.initial_alpha)
        self.beta_spin.setValue(self.initial_beta)


class NoiseReductionDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(str, int)

    def __init__(
        self,
        method: str = "Gaussian",
        ksize: int = 5,
        preview_callback: Optional[Callable[[str, int], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Noise Reduction")
        self.initial_method = method
        self.initial_ksize = ksize

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["Gaussian", "Median", "Bilateral"])
        self.method_combo.setCurrentText(method)

        self.ksize_spin = QtWidgets.QSpinBox()
        self.ksize_spin.setRange(1, 15)
        self.ksize_spin.setSingleStep(2)
        self.ksize_spin.setValue(ksize)

        form_layout.addRow("Method:", self.method_combo)
        form_layout.addRow("Kernel Size:", self.ksize_spin)

        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.method_combo.currentTextChanged.connect(self.on_value_changed)
        self.ksize_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        method, ksize = self.get_values()
        self.parametersChanged.emit(method, ksize)

    def get_values(self) -> Tuple[str, int]:
        return self.method_combo.currentText(), self.ksize_spin.value()

    def reset_to_initial(self):
        self.method_combo.setCurrentText(self.initial_method)
        self.ksize_spin.setValue(self.initial_ksize)


class SharpenDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        strength: float = 1.0,
        preview_callback: Optional[Callable[[float], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Sharpen")
        self.initial_strength = strength

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.strength_spin = QtWidgets.QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 5.0)
        self.strength_spin.setSingleStep(0.1)
        self.strength_spin.setValue(strength)

        form_layout.addRow("Strength:", self.strength_spin)

        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.strength_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        self.parametersChanged.emit(self.strength_spin.value())

    def get_value(self) -> float:
        return self.strength_spin.value()

    def reset_to_initial(self):
        self.strength_spin.setValue(self.initial_strength)


class SelectChannelDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        current_channel: str = "All",
        preview_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Select Color Channel")
        self.initial_channel = current_channel

        layout = QtWidgets.QVBoxLayout(self)

        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(["All", "R", "G", "B", "RG", "GB", "BR"])
        self.channel_combo.setCurrentText(current_channel)
        layout.addWidget(self.channel_combo)

        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.channel_combo.currentTextChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        self.parametersChanged.emit(self.channel_combo.currentText())

    def get_value(self) -> str:
        return self.channel_combo.currentText()

    def reset_to_initial(self):
        self.channel_combo.setCurrentText(self.initial_channel)


class CropDialog(QtWidgets.QDialog):
    parametersChanged = QtCore.pyqtSignal(int, int, int, int)

    def __init__(
        self,
        x_offset: int = 0,
        y_offset: int = 0,
        width: int = 100,
        height: int = 100,
        preview_callback: Optional[Callable[[int, int, int, int], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Crop")
        self.initial_x = x_offset
        self.initial_y = y_offset
        self.initial_width = width
        self.initial_height = height

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.x_spin = QtWidgets.QSpinBox()
        self.x_spin.setRange(0, 5000)
        self.x_spin.setValue(x_offset)

        self.y_spin = QtWidgets.QSpinBox()
        self.y_spin.setRange(0, 5000)
        self.y_spin.setValue(y_offset)

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 5000)
        self.width_spin.setValue(width)

        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(1, 5000)
        self.height_spin.setValue(height)

        form_layout.addRow("X Offset:", self.x_spin)
        form_layout.addRow("Y Offset:", self.y_spin)
        form_layout.addRow("Width:", self.width_spin)
        form_layout.addRow("Height:", self.height_spin)

        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.x_spin.valueChanged.connect(self.on_value_changed)
        self.y_spin.valueChanged.connect(self.on_value_changed)
        self.width_spin.valueChanged.connect(self.on_value_changed)
        self.height_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        if preview_callback is not None:
            self.parametersChanged.connect(preview_callback)

    def on_value_changed(self):
        self.parametersChanged.emit(
            self.x_spin.value(),
            self.y_spin.value(),
            self.width_spin.value(),
            self.height_spin.value(),
        )

    def get_values(self) -> Tuple[int, int, int, int]:
        return (
            self.x_spin.value(),
            self.y_spin.value(),
            self.width_spin.value(),
            self.height_spin.value(),
        )

    def reset_to_initial(self):
        self.x_spin.setValue(self.initial_x)
        self.y_spin.setValue(self.initial_y)
        self.width_spin.setValue(self.initial_width)
        self.height_spin.setValue(self.initial_height)


class MainWindow(QtWidgets.QMainWindow):
    pipelineDockVisibilityChanged = QtCore.pyqtSignal(bool)
    moduleControlsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsLoggingToggled = QtCore.pyqtSignal(bool)
    moduleActivated = QtCore.pyqtSignal(str)

    def __init__(self, app_core: AppCore):
        super().__init__()
        self.app_core = app_core
        self.setWindowTitle("Image Pre-Processing Module")
        self.resize(1200, 700)
        window_icon = load_icon(
            "manage_modules",
            fallback=self.style().standardIcon(QtWidgets.QStyle.SP_DesktopIcon),
        )
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)
        self.original_image: Optional[np.ndarray] = None
        self.base_image: Optional[np.ndarray] = None
        self.committed_image: Optional[np.ndarray] = None
        self.current_preview: Optional[np.ndarray] = None
        self.current_image_path: Optional[str] = None
        self.pipeline_manager: PipelineManager = (
            self.app_core.get_preprocessing_pipeline_manager()
        )
        self.pipeline_manager.reset()
        self.pipeline_cache = self.app_core.pipeline_cache
        self._source_id: Optional[str] = None
        self._preview_signature: Optional[str] = None
        self._committed_signature: Optional[str] = None
        self._last_pipeline_metadata: Dict[str, Any] = {}
        if self.app_core.thread_controller is None:
            self.app_core.thread_controller = ThreadController(parent=self)
        self.thread_controller: ThreadController = self.app_core.thread_controller
        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._register_thread_signals()
        # Track floating parameter editors so they can eventually be migrated into
        # docked widgets that share the asynchronous preview stream (see Issue #1).
        self._active_parameter_dialogs: Dict[str, QtWidgets.QDialog] = {}
        self._parameter_stream_sources: Dict[str, Any] = {}

        self.image_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.image_splitter.setObjectName("imageDisplaySplitter")
        self.setCentralWidget(self.image_splitter)

        original_section = SectionWidget("Original Image", self.image_splitter)
        original_section.setObjectName("originalImageSection")
        original_section.setAccessibleName("Original image panel")
        original_layout = original_section.layout
        self.original_display = ImageDisplayWidget(use_rgb_format=True)
        self.original_display.setObjectName("originalImageDisplay")
        self.original_display.setFocusPolicy(QtCore.Qt.StrongFocus)
        orig_scroll = QtWidgets.QScrollArea(original_section)
        orig_scroll.setWidgetResizable(True)
        orig_scroll.setWidget(self.original_display)
        orig_scroll.setObjectName("originalImageScrollArea")
        orig_scroll.setFocusPolicy(QtCore.Qt.StrongFocus)
        original_layout.addWidget(orig_scroll, 1)

        preview_section = SectionWidget("Pre-Processing Preview", self.image_splitter)
        preview_section.setObjectName("previewImageSection")
        preview_section.setAccessibleName("Pre-processing preview panel")
        preview_layout = preview_section.layout
        self.preview_display = ImageDisplayWidget(use_rgb_format=False)
        self.preview_display.setObjectName("previewImageDisplay")
        self.preview_display.setFocusPolicy(QtCore.Qt.StrongFocus)
        prev_scroll = QtWidgets.QScrollArea(preview_section)
        prev_scroll.setWidgetResizable(True)
        prev_scroll.setWidget(self.preview_display)
        prev_scroll.setObjectName("previewImageScrollArea")
        prev_scroll.setFocusPolicy(QtCore.Qt.StrongFocus)
        preview_layout.addWidget(prev_scroll, 1)

        self.image_splitter.addWidget(original_section)
        self.image_splitter.addWidget(preview_section)
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.setStretchFactor(0, 1)
        self.image_splitter.setStretchFactor(1, 1)

        self.pipeline_overview_list = QtWidgets.QListWidget()
        self.pipeline_overview_list.setObjectName("pipelineOverviewList")
        self.pipeline_overview_list.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.pipeline_overview_list.setAlternatingRowColors(True)
        self.pipeline_overview_list.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection
        )
        self.pipeline_overview_list.setAccessibleName("Pipeline overview")
        self.pipeline_overview_list.setWhatsThis(
            "Displays the ordered preprocessing steps currently enabled in the pipeline."
        )
        self.pipeline_overview_list.setContextMenuPolicy(
            QtCore.Qt.CustomContextMenu
        )
        self.pipeline_overview_list.customContextMenuRequested.connect(
            self._show_module_context_menu
        )

        pipeline_widget = SectionWidget("Pipeline Summary")
        pipeline_widget.setObjectName("pipelineSummarySection")
        pipeline_layout = pipeline_widget.layout
        self.pipeline_label = QtWidgets.QLabel("Current Pipeline: (none)")
        self.pipeline_label.setObjectName("pipelineSummaryLabel")
        self.pipeline_label.setWordWrap(True)
        self.pipeline_label.setAccessibleDescription(
            "Text summary of the current preprocessing pipeline order."
        )
        pipeline_layout.addWidget(self.pipeline_label)
        pipeline_layout.addWidget(self.pipeline_overview_list, 1)

        self.pipeline_dock = ThemedDockWidget("Pipeline Overview", self)
        self.pipeline_dock.setObjectName("pipelineOverviewDock")
        self.pipeline_dock.setWidget(pipeline_widget)
        self.pipeline_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.pipeline_dock.setToolTip("Pipeline overview panel (Ctrl+1)")
        self.pipeline_dock.setWhatsThis(
            "Dockable panel summarizing the active preprocessing pipeline."
        )
        self.pipeline_dock.setAccessibleName("Pipeline overview dock")
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.pipeline_dock)
        self.pipeline_dock.visibilityChanged.connect(
            self._on_pipeline_dock_visibility_changed
        )

        self.diagnostics_log = QtWidgets.QPlainTextEdit()
        self.diagnostics_log.setReadOnly(True)
        self.diagnostics_log.setObjectName("diagnosticsLog")
        self.diagnostics_log.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.diagnostics_log.setAccessibleName("Diagnostics log viewer")
        self.diagnostics_log.setPlaceholderText(
            "Diagnostics output will appear here when verbose logging is enabled."
        )
        self.diagnostics_log.setWhatsThis(
            "Displays diagnostic messages and progress updates emitted during preprocessing."
        )

        diagnostics_widget = SectionWidget("Diagnostics & Shortcuts")
        diagnostics_widget.setObjectName("diagnosticsSection")
        diagnostics_layout = diagnostics_widget.layout
        self.shortcut_summary = ShortcutSummaryWidget()
        diagnostics_layout.addWidget(self.shortcut_summary)
        diagnostics_layout.addWidget(self.diagnostics_log, 1)

        self.diagnostics_dock = ThemedDockWidget("Diagnostics Log", self)
        self.diagnostics_dock.setObjectName("diagnosticsLogDock")
        self.diagnostics_dock.setWidget(diagnostics_widget)
        self.diagnostics_dock.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea
            | QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
        )
        self.diagnostics_dock.setToolTip("Diagnostics log panel (Ctrl+2)")
        self.diagnostics_dock.setWhatsThis(
            "Dockable panel displaying diagnostic output and log messages."
        )
        self.diagnostics_dock.setAccessibleName("Diagnostics log dock")
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.diagnostics_dock)
        self.diagnostics_dock.visibilityChanged.connect(
            self._on_diagnostics_dock_visibility_changed
        )

        self.module_controls_container = QtWidgets.QScrollArea()
        self.module_controls_container.setWidgetResizable(True)
        self.module_controls_container.setObjectName("moduleControlsScrollArea")
        self.module_controls_container.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.module_controls_container.setAccessibleName("Module parameter controls")
        self.module_controls_container.setWhatsThis(
            "Holds parameter controls for the selected preprocessing module."
        )
        module_controls_content = SectionWidget("Module Parameters")
        module_controls_content.setObjectName("moduleControlsSection")
        self.module_controls_layout = module_controls_content.layout
        self.module_controls_placeholder = QtWidgets.QLabel(
            "Select a module to configure its parameters."
        )
        self.module_controls_placeholder.setWordWrap(True)
        self.module_controls_placeholder.setFont(
            scale_font(self.font(), factor=1.0)
        )
        self.module_controls_layout.addWidget(self.module_controls_placeholder)
        self.module_controls_layout.addStretch(1)
        self.module_controls_container.setWidget(module_controls_content)

        self.module_controls_dock = ThemedDockWidget(
            "Module Parameters", self
        )
        self.module_controls_dock.setObjectName("moduleParametersDock")
        self.module_controls_dock.setWidget(self.module_controls_container)
        self.module_controls_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.module_controls_dock.setToolTip("Module parameter controls (Ctrl+3)")
        self.module_controls_dock.setWhatsThis(
            "Dockable panel hosting the configuration widgets for preprocessing modules."
        )
        self.module_controls_dock.setAccessibleName("Module parameter dock")
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.module_controls_dock)
        self.module_controls_dock.visibilityChanged.connect(
            self._on_module_controls_dock_visibility_changed
        )

        self.shortcut_status_label = QtWidgets.QLabel()
        self.shortcut_status_label.setObjectName("shortcutStatusLabel")
        self.shortcut_status_label.setAccessibleName(
            "Primary keyboard shortcuts summary"
        )
        self.statusBar().addPermanentWidget(self.shortcut_status_label, 1)
        self.shortcut_registry = ShortcutRegistry(
            summary_widget=self.shortcut_summary,
            status_label=self.shortcut_status_label,
            parent=self,
        )

        self._module_action_lookup: Dict[
            Tuple[str, str, Tuple[str, ...]], QtWidgets.QAction
        ] = {}

        self._pipeline_focus_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+1"), self
        )
        self._pipeline_focus_shortcut.setObjectName("pipelineFocusShortcut")
        self._pipeline_focus_shortcut.activated.connect(
            lambda: self.pipeline_overview_list.setFocus(QtCore.Qt.ShortcutFocusReason)
        )

        self._diagnostics_focus_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+2"), self
        )
        self._diagnostics_focus_shortcut.setObjectName("diagnosticsFocusShortcut")
        self._diagnostics_focus_shortcut.activated.connect(
            lambda: self.diagnostics_log.setFocus(QtCore.Qt.ShortcutFocusReason)
        )

        self._module_controls_focus_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+3"), self
        )
        self._module_controls_focus_shortcut.setObjectName(
            "moduleControlsFocusShortcut"
        )
        self._module_controls_focus_shortcut.activated.connect(
            lambda: self.module_controls_container.setFocus(
                QtCore.Qt.ShortcutFocusReason
            )
        )

        self.statusBar().showMessage("Ready")

        self.build_menu()

        self.undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.redo)
        self._register_static_shortcuts()

        self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
        self.update_pipeline_label()
        if self.base_image is not None:
            self.update_preview()

    def update_pipeline_label(self):
        order = [step.name for step in self.pipeline_manager.iter_enabled_steps()]
        text = "Current Pipeline: " + " -> ".join(order) if order else "Current Pipeline: (none)"
        self.pipeline_label.setText(text)
        self.pipeline_overview_list.clear()
        if order:
            self.pipeline_overview_list.addItems(order)
        else:
            self.pipeline_overview_list.addItem("(no enabled steps)")

    def rebuild_pipeline(self):
        self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
        logging.debug("Pipeline rebuilt with steps: " + ", ".join(step.name for step in self.pipeline.steps))
        self.update_pipeline_label()

    def undo(self):
        snapshot = self.pipeline_manager.undo(
            current_image=self.committed_image,
            current_cache_signature=self._committed_signature,
        )
        if snapshot is None:
            return
        self._committed_signature = snapshot.cache_signature
        cached_image: Optional[np.ndarray] = None
        metadata: Dict[str, Any] = {}
        if snapshot.cache_signature and self._source_id is not None:
            cached_image = self.pipeline_cache.get_cached_image(
                self._source_id, snapshot.cache_signature
            )
            metadata = self.pipeline_cache.metadata_for(
                self._source_id, snapshot.cache_signature
            )
        image_to_use: Optional[np.ndarray] = None
        if cached_image is not None:
            image_to_use = cached_image
        elif snapshot.image is not None:
            image_to_use = snapshot.image.copy()

        self.rebuild_pipeline()

        need_compute = False
        if image_to_use is not None:
            self.committed_image = np.array(image_to_use, copy=True)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
            self._preview_signature = snapshot.cache_signature
            if metadata:
                self._last_pipeline_metadata = metadata
            else:
                self._last_pipeline_metadata = {}
                need_compute = True
        else:
            self._last_pipeline_metadata = {}
            self._preview_signature = snapshot.cache_signature
            need_compute = True

        if need_compute and self.base_image is not None:
            callback = self._update_committed_from_result
            self._apply_pipeline_async(
                description="Updating preview",
                on_finished=callback,
            )
        self.update_undo_redo_actions()

    def redo(self):
        snapshot = self.pipeline_manager.redo(
            current_image=self.committed_image,
            current_cache_signature=self._committed_signature,
        )
        if snapshot is None:
            return
        self._committed_signature = snapshot.cache_signature
        cached_image: Optional[np.ndarray] = None
        metadata: Dict[str, Any] = {}
        if snapshot.cache_signature and self._source_id is not None:
            cached_image = self.pipeline_cache.get_cached_image(
                self._source_id, snapshot.cache_signature
            )
            metadata = self.pipeline_cache.metadata_for(
                self._source_id, snapshot.cache_signature
            )
        image_to_use: Optional[np.ndarray] = None
        if cached_image is not None:
            image_to_use = cached_image
        elif snapshot.image is not None:
            image_to_use = snapshot.image.copy()

        self.rebuild_pipeline()

        need_compute = False
        if image_to_use is not None:
            self.committed_image = np.array(image_to_use, copy=True)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
            self._preview_signature = snapshot.cache_signature
            if metadata:
                self._last_pipeline_metadata = metadata
            else:
                self._last_pipeline_metadata = {}
                need_compute = True
        else:
            self._last_pipeline_metadata = {}
            self._preview_signature = snapshot.cache_signature
            need_compute = True

        if need_compute and self.base_image is not None:
            callback = self._update_committed_from_result
            self._apply_pipeline_async(
                description="Updating preview",
                on_finished=callback,
            )
        self.update_undo_redo_actions()

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(self.pipeline_manager.can_undo())
        self.redo_action.setEnabled(self.pipeline_manager.can_redo())

    def build_menu(self):
        menubar = self.menuBar()
        menubar.clear()

        if hasattr(self, "shortcut_registry"):
            self.shortcut_registry.reset()

        menu_cache: Dict[Tuple[str, ...], QtWidgets.QMenu] = {}
        self._module_action_lookup.clear()

        def ensure_menu(path: Tuple[str, ...]) -> QtWidgets.QMenu:
            if path in menu_cache:
                return menu_cache[path]
            if not path:
                raise ValueError("Menu path cannot be empty")
            if len(path) == 1:
                menu = menubar.addMenu(path[0])
            else:
                parent = ensure_menu(path[:-1])
                menu = parent.addMenu(path[-1])
            menu_cache[path] = menu
            return menu

        file_menu = ensure_menu(("File",))

        self.load_image_action = QtWidgets.QAction("&Load Image...", self)
        self.load_image_action.setShortcut(QtGui.QKeySequence.Open)
        self.load_image_action.setShortcutVisibleInContextMenu(True)
        self.load_image_action.setStatusTip("Load an image for preprocessing")
        self.load_image_action.triggered.connect(self.load_image)
        self.load_image_action.setIcon(
            load_icon(
                "open_project",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton),
            )
        )
        file_menu.addAction(self.load_image_action)
        self.shortcut_registry.register_action("Load image", self.load_image_action)

        self.save_processed_image_action = QtWidgets.QAction(
            "&Save Pre-Processed Image...", self
        )
        self.save_processed_image_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_processed_image_action.setShortcutVisibleInContextMenu(True)
        self.save_processed_image_action.setStatusTip(
            "Persist the currently committed preprocessing result"
        )
        self.save_processed_image_action.triggered.connect(
            self.save_processed_image
        )
        self.save_processed_image_action.setIcon(
            load_icon(
                "save_project",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            )
        )
        file_menu.addAction(self.save_processed_image_action)
        self.shortcut_registry.register_action(
            "Save pre-processed image", self.save_processed_image_action
        )

        file_menu.addSeparator()

        self.mass_preprocess_action = QtWidgets.QAction(
            "Mass Pre-Process &Folder...", self
        )
        self.mass_preprocess_action.setShortcut("Ctrl+Shift+M")
        self.mass_preprocess_action.setShortcutVisibleInContextMenu(True)
        self.mass_preprocess_action.setStatusTip(
            "Apply the active pipeline to every image in a folder"
        )
        self.mass_preprocess_action.triggered.connect(self.mass_preprocess)
        self.mass_preprocess_action.setIcon(
            load_icon(
                "manage_modules",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
            )
        )
        file_menu.addAction(self.mass_preprocess_action)
        self.shortcut_registry.register_action(
            "Mass pre-process folder", self.mass_preprocess_action
        )

        file_menu.addSeparator()

        self.import_pipeline_action = QtWidgets.QAction(
            "&Import Pipeline Settings...", self
        )
        self.import_pipeline_action.setShortcut("Ctrl+I")
        self.import_pipeline_action.setShortcutVisibleInContextMenu(True)
        self.import_pipeline_action.setStatusTip(
            "Load preprocessing pipeline settings from disk"
        )
        self.import_pipeline_action.triggered.connect(self.import_pipeline)
        self.import_pipeline_action.setIcon(
            load_icon(
                "open_project",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton),
            )
        )
        file_menu.addAction(self.import_pipeline_action)
        self.shortcut_registry.register_action(
            "Import pipeline settings", self.import_pipeline_action
        )

        self.export_pipeline_action = QtWidgets.QAction(
            "E&xport Pipeline Settings...", self
        )
        self.export_pipeline_action.setShortcut("Ctrl+Shift+E")
        self.export_pipeline_action.setShortcutVisibleInContextMenu(True)
        self.export_pipeline_action.setStatusTip(
            "Save preprocessing pipeline settings to disk"
        )
        self.export_pipeline_action.triggered.connect(self.export_pipeline)
        self.export_pipeline_action.setIcon(
            load_icon(
                "save_project_as",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            )
        )
        file_menu.addAction(self.export_pipeline_action)
        self.shortcut_registry.register_action(
            "Export pipeline settings", self.export_pipeline_action
        )

        edit_menu = ensure_menu(("Edit",))
        self.undo_action = QtWidgets.QAction("&Undo", self)
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)
        self.undo_action.setShortcutVisibleInContextMenu(True)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        self.undo_action.setIcon(
            load_icon(
                "undo",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack),
            )
        )
        edit_menu.addAction(self.undo_action)
        self.shortcut_registry.register_action("Undo", self.undo_action)

        self.redo_action = QtWidgets.QAction("&Redo", self)
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)
        self.redo_action.setShortcutVisibleInContextMenu(True)
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        self.redo_action.setIcon(
            load_icon(
                "redo",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward),
            )
        )
        edit_menu.addAction(self.redo_action)
        self.shortcut_registry.register_action("Redo", self.redo_action)

        edit_menu.addSeparator()

        self.reset_pipeline_action = QtWidgets.QAction("Reset &All", self)
        self.reset_pipeline_action.setShortcut("Ctrl+Shift+R")
        self.reset_pipeline_action.setShortcutVisibleInContextMenu(True)
        self.reset_pipeline_action.setStatusTip(
            "Restore the preprocessing pipeline to default settings"
        )
        self.reset_pipeline_action.triggered.connect(self.reset_all)
        self.reset_pipeline_action.setIcon(
            load_icon(
                "redo",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload),
            )
        )
        edit_menu.addAction(self.reset_pipeline_action)
        self.shortcut_registry.register_action(
            "Reset pipeline", self.reset_pipeline_action
        )

        view_menu = ensure_menu(("View",))

        self.show_pipeline_dock_action = QtWidgets.QAction(
            "Pipeline Overview", self
        )
        self.show_pipeline_dock_action.setCheckable(True)
        self.show_pipeline_dock_action.setChecked(self.pipeline_dock.isVisible())
        self.show_pipeline_dock_action.setShortcut("Alt+1")
        self.show_pipeline_dock_action.setStatusTip(
            "Show or hide the pipeline overview dock"
        )
        self.show_pipeline_dock_action.toggled.connect(
            self.pipeline_dock.setVisible
        )
        self.show_pipeline_dock_action.setIcon(
            load_icon(
                "manage_modules",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView),
            )
        )
        view_menu.addAction(self.show_pipeline_dock_action)
        self.shortcut_registry.register_action(
            "Toggle pipeline overview", self.show_pipeline_dock_action
        )

        self.show_diagnostics_dock_action = QtWidgets.QAction(
            "Diagnostics Log", self
        )
        self.show_diagnostics_dock_action.setCheckable(True)
        self.show_diagnostics_dock_action.setChecked(
            self.diagnostics_dock.isVisible()
        )
        self.show_diagnostics_dock_action.setShortcut("Alt+2")
        self.show_diagnostics_dock_action.setStatusTip(
            "Show or hide the diagnostics log dock"
        )
        self.show_diagnostics_dock_action.toggled.connect(
            self.diagnostics_dock.setVisible
        )
        self.show_diagnostics_dock_action.setIcon(
            load_icon(
                "documentation",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView),
            )
        )
        view_menu.addAction(self.show_diagnostics_dock_action)
        self.shortcut_registry.register_action(
            "Toggle diagnostics log", self.show_diagnostics_dock_action
        )

        self.show_module_controls_dock_action = QtWidgets.QAction(
            "Module Controls", self
        )
        self.show_module_controls_dock_action.setCheckable(True)
        self.show_module_controls_dock_action.setChecked(
            self.module_controls_dock.isVisible()
        )
        self.show_module_controls_dock_action.setShortcut("Alt+3")
        self.show_module_controls_dock_action.setStatusTip(
            "Show or hide the module parameter controls dock"
        )
        self.show_module_controls_dock_action.toggled.connect(
            self.module_controls_dock.setVisible
        )
        self.show_module_controls_dock_action.setIcon(
            load_icon(
                "manage_modules",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
            )
        )
        view_menu.addAction(self.show_module_controls_dock_action)
        self.shortcut_registry.register_action(
            "Toggle module controls", self.show_module_controls_dock_action
        )

        view_menu.addSeparator()

        self.enable_diagnostics_logging_action = QtWidgets.QAction(
            "Enable Diagnostics Logging", self
        )
        self.enable_diagnostics_logging_action.setCheckable(True)
        self.enable_diagnostics_logging_action.setChecked(
            self.app_core.diagnostics_enabled
        )
        self.enable_diagnostics_logging_action.setStatusTip(
            "Toggle verbose diagnostics logging for troubleshooting"
        )
        self.enable_diagnostics_logging_action.toggled.connect(
            self._set_diagnostics_logging
        )
        view_menu.addAction(self.enable_diagnostics_logging_action)

        ensure_menu(("Modules",))

        for module in self.app_core.get_modules(ModuleStage.PREPROCESSING):
            for entry in module.menu_entries():
                path = entry.path
                if not path or path[0] != "Modules":
                    path = ("Modules", *path)
                menu = ensure_menu(path)
                action = QtWidgets.QAction(entry.text, self)
                action.setObjectName(
                    f"moduleAction_{module.metadata.identifier}_{'_'.join(entry.path)}"
                )
                if entry.shortcut:
                    action.setShortcut(entry.shortcut)
                    action.setShortcutVisibleInContextMenu(True)
                description = entry.description or module.metadata.description
                if description:
                    action.setStatusTip(description)
                action.triggered.connect(
                    partial(self._on_module_action_triggered, module)
                )
                menu.addAction(action)
                key = (module.metadata.identifier, entry.text, entry.path)
                self._module_action_lookup[key] = action
                self.shortcut_registry.register_action(entry.text, action)

        help_menu = ensure_menu(("Help",))

        self.view_documentation_action = QtWidgets.QAction(
            "View &Documentation", self
        )
        self.view_documentation_action.setShortcut(
            QtGui.QKeySequence.HelpContents
        )
        self.view_documentation_action.setStatusTip(
            "Open the user documentation for Yam Image Processor"
        )
        self.view_documentation_action.triggered.connect(self.show_documentation)
        self.view_documentation_action.setIcon(
            load_icon(
                "documentation",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_DialogHelpButton),
            )
        )
        help_menu.addAction(self.view_documentation_action)
        self.shortcut_registry.register_action(
            "View documentation", self.view_documentation_action
        )

        self.about_action = QtWidgets.QAction("&About", self)
        self.about_action.setStatusTip("Show application information")
        self.about_action.triggered.connect(
            lambda: QtWidgets.QMessageBox.information(
                self,
                "About",
                "Yam Image Processor Pre-Processing module",
            )
        )
        self.about_action.setIcon(
            load_icon(
                "about",
                fallback=self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView),
            )
        )
        help_menu.addAction(self.about_action)
        self.shortcut_registry.register_action("About", self.about_action)

        self.update_undo_redo_actions()
        self._register_static_shortcuts()

    def _register_static_shortcuts(self) -> None:
        if not hasattr(self, "shortcut_registry"):
            return
        mapping = (
            ("Focus pipeline overview", "_pipeline_focus_shortcut"),
            ("Focus diagnostics log", "_diagnostics_focus_shortcut"),
            ("Focus module controls", "_module_controls_focus_shortcut"),
            ("Undo", "undo_shortcut"),
            ("Redo", "redo_shortcut"),
        )
        for label, attr in mapping:
            shortcut = getattr(self, attr, None)
            if shortcut is not None:
                self.shortcut_registry.register_shortcut(label, shortcut)

    def _activate_module(self, module: ModuleBase) -> None:
        try:
            module.activate(self)
            self.moduleActivated.emit(module.metadata.identifier)
        except NotImplementedError:
            logging.warning(
                "Module %s does not implement an activation handler",
                module.metadata.identifier,
            )
            self.statusBar().showMessage(
                f"{module.metadata.title} is not available for activation.", 2000
            )
        except Exception as exc:  # pragma: no cover - defensive UI guard
            logging.exception("Module activation failed: %%s", module.metadata.identifier)
            QtWidgets.QMessageBox.critical(
                self,
                "Module Error",
                f"{module.metadata.title} failed to run.\n{exc}",
            )
            self.statusBar().showMessage("Error running module action", 4000)

    def _register_thread_signals(self) -> None:
        self._current_task_description: str = ""
        self.thread_controller.task_started.connect(self._on_task_started)
        self.thread_controller.task_progress.connect(self._on_task_progress)
        self.thread_controller.task_finished.connect(self._on_task_finished)
        self.thread_controller.task_canceled.connect(self._on_task_canceled)
        self.thread_controller.task_failed.connect(self._on_task_failed)

    def _on_pipeline_dock_visibility_changed(self, visible: bool) -> None:
        if hasattr(self, "show_pipeline_dock_action"):
            self.show_pipeline_dock_action.blockSignals(True)
            self.show_pipeline_dock_action.setChecked(visible)
            self.show_pipeline_dock_action.blockSignals(False)
        self.pipelineDockVisibilityChanged.emit(visible)

    def _on_module_controls_dock_visibility_changed(self, visible: bool) -> None:
        if hasattr(self, "show_module_controls_dock_action"):
            self.show_module_controls_dock_action.blockSignals(True)
            self.show_module_controls_dock_action.setChecked(visible)
            self.show_module_controls_dock_action.blockSignals(False)
        self.moduleControlsDockVisibilityChanged.emit(visible)

    def _on_diagnostics_dock_visibility_changed(self, visible: bool) -> None:
        if hasattr(self, "show_diagnostics_dock_action"):
            self.show_diagnostics_dock_action.blockSignals(True)
            self.show_diagnostics_dock_action.setChecked(visible)
            self.show_diagnostics_dock_action.blockSignals(False)
        self.diagnosticsDockVisibilityChanged.emit(visible)

    def _set_diagnostics_logging(self, enabled: bool) -> None:
        self.app_core.set_diagnostics_enabled(enabled)
        message = (
            "Diagnostics logging enabled."
            if enabled
            else "Diagnostics logging disabled."
        )
        self._append_diagnostic_message(message)
        self.statusBar().showMessage(message, 3000)
        self.diagnosticsLoggingToggled.emit(enabled)

    def _on_module_action_triggered(self, module: ModuleBase) -> None:
        self._activate_module(module)

    def _show_module_context_menu(self, position: QtCore.QPoint) -> None:
        if not self._module_action_lookup:
            return
        menu = QtWidgets.QMenu(self.pipeline_overview_list)
        actions = sorted(
            self._module_action_lookup.values(), key=lambda action: action.text().lower()
        )
        for action in actions:
            menu.addAction(action)
        menu.exec_(self.pipeline_overview_list.mapToGlobal(position))

    def show_documentation(self) -> None:
        docs_root = Path(__file__).resolve().parents[1] / "docs"
        candidates = [
            docs_root / "USER_GUIDE.md",
            docs_root / "README.md",
            Path(__file__).resolve().parents[1] / "README.md",
        ]
        for candidate in candidates:
            if candidate.exists():
                url = QtCore.QUrl.fromLocalFile(str(candidate))
                QtGui.QDesktopServices.openUrl(url)
                self.statusBar().showMessage(
                    f"Opened documentation: {candidate.name}", 3000
                )
                return
        QtWidgets.QMessageBox.information(
            self,
            "Documentation",
            "Documentation files are not available in this build.",
        )

    def _append_diagnostic_message(self, message: str) -> None:
        if not hasattr(self, "diagnostics_log") or self.diagnostics_log is None:
            return
        timestamp = QtCore.QDateTime.currentDateTime().toString(
            "yyyy-MM-dd hh:mm:ss.zzz"
        )
        self.diagnostics_log.appendPlainText(f"[{timestamp}] {message}")

    @QtCore.pyqtSlot(str)
    def _on_task_started(self, description: str) -> None:
        self._current_task_description = description or "Processing"
        self.statusBar().showMessage(self._current_task_description)
        self._append_diagnostic_message(f"{self._current_task_description} started.")

    @QtCore.pyqtSlot(int)
    def _on_task_progress(self, value: int) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.setValue(value)

    @QtCore.pyqtSlot(object)
    def _on_task_finished(self, _result: object) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        self.statusBar().showMessage("Ready", 1500)
        self._append_diagnostic_message(
            f"{self._current_task_description or 'Processing'} completed successfully."
        )

    @QtCore.pyqtSlot()
    def _on_task_canceled(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        self.statusBar().showMessage("Operation canceled", 2000)
        self._append_diagnostic_message(
            f"{self._current_task_description or 'Processing'} was canceled."
        )

    @QtCore.pyqtSlot(Exception, str)
    def _on_task_failed(self, error: Exception, stack: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        logging.error("Pipeline execution failed: %s\n%s", error, stack)
        self._append_diagnostic_message(
            f"{self._current_task_description or 'Processing'} failed: {error}"
        )
        QtWidgets.QMessageBox.critical(
            self,
            "Processing Error",
            f"{self._current_task_description} failed:\n{error}",
        )
        self.statusBar().showMessage("Error during processing", 4000)

    def _ensure_source_registered(self, image: np.ndarray) -> Optional[str]:
        if self._source_id is None:
            hint = self.current_image_path or None
            self._source_id = self.pipeline_cache.register_source(image, hint=hint)
        return self._source_id

    def _apply_pipeline_async(
        self,
        *,
        pipeline: Optional[PipelineManager] = None,
        image: Optional[np.ndarray] = None,
        description: str = "Processing image",
        on_finished: Optional[Callable[[PipelineCacheResult], None]] = None,
        on_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        if pipeline is None:
            pipeline = self.pipeline
        if pipeline is None:
            return
        if image is None:
            if self.base_image is None:
                return
            image = self.base_image

        source_id = self._ensure_source_registered(image)
        if source_id is None:
            return

        steps = tuple(step.clone() for step in pipeline.steps)

        def _handle_finished(result: PipelineCacheResult) -> None:
            if on_finished is not None:
                on_finished(result)

        def _handle_canceled() -> None:
            if on_canceled is not None:
                on_canceled()

        def _task(cancel_event: threading.Event, progress: Callable[[int], None]):
            return self.pipeline_cache.compute(
                source_id=source_id,
                image=image,
                steps=steps,
                cancel_event=cancel_event,
                progress=progress,
            )

        self.thread_controller.run_task(
            _task,
            description=description,
            on_finished=_handle_finished,
            on_canceled=_handle_canceled,
        )

    def _present_parameter_dialog(
        self,
        key: str,
        dialog: QtWidgets.QDialog,
        *,
        on_accept: Callable[[], None],
        on_cancel: Optional[Callable[[], None]] = None,
        on_parameters_changed: Optional[Callable[..., None]] = None,
    ) -> None:
        existing = self._active_parameter_dialogs.pop(key, None)
        if existing is not None:
            existing.close()
        dialog.setModal(True)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        if on_parameters_changed is not None and hasattr(dialog, "parametersChanged"):
            dialog.parametersChanged.connect(on_parameters_changed)  # type: ignore[attr-defined]
            self._parameter_stream_sources[key] = dialog.parametersChanged  # type: ignore[attr-defined]
        dialog.accepted.connect(on_accept)
        if on_cancel is not None:
            dialog.rejected.connect(on_cancel)

        def _cleanup(_result: int) -> None:
            self._active_parameter_dialogs.pop(key, None)
            self._parameter_stream_sources.pop(key, None)

        dialog.finished.connect(_cleanup)
        self._active_parameter_dialogs[key] = dialog
        dialog.open()

    def _update_preview_from_result(self, result: PipelineCacheResult) -> None:
        self._preview_signature = result.final_signature
        self.current_preview = result.image.copy()
        self.preview_display.set_image(self.current_preview)

    def _update_committed_from_result(self, result: PipelineCacheResult) -> None:
        self.committed_image = result.image.copy()
        self.current_preview = self.committed_image.copy()
        self.preview_display.set_image(self.current_preview)
        self._committed_signature = result.final_signature
        self._preview_signature = result.final_signature
        self._last_pipeline_metadata = result.metadata
        self.update_undo_redo_actions()
        try:
            autosave = self.app_core.autosave
        except RuntimeError:
            autosave = None
        if autosave is not None and self.committed_image is not None:
            pipeline_payload = {"stage": "preprocessing", **self.pipeline_manager.to_dict()}
            pipeline_payload["cache_signature"] = result.final_signature
            metadata: Dict[str, Any] = {
                "stage": "preprocessing",
                "mode": "single",
                "cache": result.metadata,
            }
            if self.current_image_path:
                metadata.setdefault("source", {})["input"] = self.current_image_path
            autosave.mark_dirty(self.committed_image, pipeline_payload, metadata=metadata)

    def reset_all(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        self.pipeline_manager.push_state(
            image=backup, cache_signature=self._committed_signature
        )
        self.pipeline_manager.replace_steps(
            self.pipeline_manager.template_steps(), preserve_history=True
        )
        self.rebuild_pipeline()
        if self.base_image is not None:
            self._apply_pipeline_async(
                description="Resetting pipeline",
                on_finished=self._on_reset_completed,
            )
        self.update_undo_redo_actions()

    def _on_reset_completed(self, result: PipelineCacheResult) -> None:
        self._update_committed_from_result(result)
        self.statusBar().showMessage("Reset all processing to defaults.", 3000)

    def mass_preprocess(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder for Mass Pre-Process")
        if not folder:
            return
        parent_dir = os.path.dirname(folder)
        base_folder = os.path.basename(folder)
        output_folder = os.path.join(parent_dir, base_folder + "_pp")
        os.makedirs(output_folder, exist_ok=True)
        files = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, file))
            and os.path.splitext(file)[1].lower() in Config.SUPPORTED_FORMATS
        ]
        if not files:
            QtWidgets.QMessageBox.information(
                self,
                "Mass Pre-Process",
                "No supported images were found in the selected folder.",
            )
            return

        self._progress_dialog = QtWidgets.QProgressDialog(
            "Processing images…",
            "Cancel",
            0,
            100,
            self,
        )
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setValue(0)
        self._progress_dialog.canceled.connect(self.thread_controller.cancel)

        pipeline_snapshot = build_preprocessing_pipeline(
            self.app_core, self.pipeline_manager.clone()
        )
        pipeline_metadata = {"stage": "preprocessing", **pipeline_snapshot.to_dict()}
        settings_snapshot = self.app_core.settings.snapshot(prefix="preprocess/")
        io_manager = self.app_core.io_manager

        def _task(cancel_event: threading.Event, progress: Callable[[int], None]) -> int:
            processed_count = 0
            total = len(files)
            for index, path in enumerate(files, start=1):
                if cancel_event.is_set():
                    raise OperationCancelled()
                filename = os.path.basename(path)
                try:
                    image = Loader.load_image(path)
                    result = pipeline_snapshot.apply(image)
                    name, ext = os.path.splitext(filename)
                    new_filename = name + "_pp" + ext
                    outpath = os.path.join(output_folder, new_filename)
                    io_manager.save_image(
                        outpath,
                        result,
                        metadata={
                            "stage": "preprocessing",
                            "mode": "batch",
                            "source": {
                                "input": path,
                                "index": index,
                                "total": total,
                            },
                        },
                        pipeline=pipeline_metadata,
                        settings_snapshot=settings_snapshot,
                    )
                    processed_count += 1
                except Exception as exc:  # pragma: no cover - user feedback
                    logging.error("Failed to process %s: %s", filename, exc)
                finally:
                    progress(int(index * 100 / total))
            return processed_count

        def _finished(count: int) -> None:
            QtWidgets.QMessageBox.information(
                self,
                "Mass Pre-Process",
                f"Processed {count} images.\nOutput folder: {output_folder}",
            )

        def _canceled() -> None:
            QtWidgets.QMessageBox.information(
                self,
                "Mass Pre-Process",
                "Mass pre-processing was canceled.",
            )

        self.thread_controller.run_task(
            _task,
            description="Mass pre-processing images",
            on_finished=_finished,
            on_canceled=_canceled,
        )

    def export_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Pipeline Settings", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                payload = json.dumps(self.pipeline_manager.to_dict(), indent=2)
                Path(filename).write_text(payload, encoding="utf-8")
                QtWidgets.QMessageBox.information(self, "Pipeline Export", "Pipeline settings exported.")
            except Exception as exc:  # pragma: no cover - user feedback
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export pipeline: {exc}")

    def import_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Pipeline Settings", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                payload = Path(filename).read_text(encoding="utf-8")
                data = json.loads(payload)
                self.pipeline_manager.push_state(
                    image=None if self.committed_image is None else self.committed_image.copy(),
                    cache_signature=self._committed_signature,
                )
                self.app_core.load_preprocessing_pipeline(data)
                self.rebuild_pipeline()
                if self.base_image is not None:
                    def _applied(result: PipelineCacheResult) -> None:
                        self._update_committed_from_result(result)
                        QtWidgets.QMessageBox.information(
                            self,
                            "Pipeline Import",
                            "Pipeline settings imported and applied.",
                        )

                    self._apply_pipeline_async(
                        description="Applying imported pipeline",
                        on_finished=_applied,
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Pipeline Import",
                        "Pipeline settings imported.",
                    )
                self.update_undo_redo_actions()
            except Exception as exc:  # pragma: no cover - user feedback
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to import pipeline: {exc}")

    def load_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.jpg *.png *.tiff *.bmp *.npy)"
        )
        if not filename:
            return
        try:
            image = Loader.load_image(filename)
        except Exception as exc:  # pragma: no cover - user feedback
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {exc}")
            return

        if self._source_id is not None:
            self.pipeline_cache.discard_cache(self._source_id)
        self.original_image = image.copy()
        self.base_image = image.copy()
        self.committed_image = image.copy()
        self.current_image_path = filename
        self._source_id = self.pipeline_cache.register_source(self.base_image, hint=filename)
        self._committed_signature = self._source_id
        self._preview_signature = self._source_id
        self._last_pipeline_metadata = self.pipeline_cache.metadata_for(
            self._source_id, self._source_id
        )
        self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
        self.original_display.set_image(self.original_image)
        self.preview_display.set_image(self.base_image)
        self.update_preview()
        self.pipeline_manager.clear_history()
        self.update_undo_redo_actions()
        self.statusBar().showMessage(f"Loaded image: {filename}")

    def save_processed_image(self):
        if self.committed_image is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "No processed image to save.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Pre-Processed Image", "", "Image Files (*.jpg *.png *.tiff *.bmp *.npy)"
        )
        if filename:
            io_manager = self.app_core.io_manager
            pipeline_metadata = {"stage": "preprocessing", **self.pipeline_manager.to_dict()}
            if self._committed_signature is not None:
                pipeline_metadata["cache_signature"] = self._committed_signature
            settings_snapshot = self.app_core.settings.snapshot(prefix="preprocess/")
            metadata: Dict[str, Any] = {
                "stage": "preprocessing",
                "mode": "single",
            }
            if self.current_image_path:
                metadata["source"] = {"input": self.current_image_path}
            if self._committed_signature is not None:
                metadata["result_signature"] = self._committed_signature
            if self._last_pipeline_metadata:
                metadata["cache"] = self._last_pipeline_metadata
            result = io_manager.save_image(
                filename,
                self.committed_image,
                metadata=metadata,
                pipeline=pipeline_metadata,
                settings_snapshot=settings_snapshot,
            )
            self.statusBar().showMessage(f"Saved processed image to: {result.image_path}")

    def update_preview(self):
        if self.base_image is None or self._source_id is None or self.pipeline is None:
            return
        steps = tuple(step.clone() for step in self.pipeline.steps)
        final_signature, _ = self.pipeline_cache.predict(self._source_id, steps)
        cached = self.pipeline_cache.get_cached_image(self._source_id, final_signature)
        if cached is not None:
            self._preview_signature = final_signature
            self.current_preview = cached.copy()
            self.preview_display.set_image(self.current_preview)
            return
        self._apply_pipeline_async(
            description="Updating preview",
            on_finished=self._update_preview_from_result,
        )

    def toggle_grayscale(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        self.pipeline_manager.push_state(
            image=backup, cache_signature=self._committed_signature
        )
        self.pipeline_manager.toggle_step("Grayscale")
        self.rebuild_pipeline()
        if self.base_image is not None:
            self._apply_pipeline_async(
                description="Applying grayscale step",
                on_finished=self._update_committed_from_result,
            )
        self.update_undo_redo_actions()

    def preview_update(self, func_name: str, temp_params: Dict[str, Any]):
        if self.base_image is None or self._source_id is None:
            return
        temp_manager = self.pipeline_manager.clone()
        try:
            step = temp_manager.get_step(func_name)
        except KeyError:
            return
        step.enabled = True
        step.params.update(temp_params)
        if func_name == "Crop":
            step.params["apply_crop"] = False
        steps = tuple(step.clone() for step in temp_manager.steps)
        final_signature, _ = self.pipeline_cache.predict(self._source_id, steps)
        cached = self.pipeline_cache.get_cached_image(self._source_id, final_signature)
        if cached is not None:
            self._preview_signature = final_signature
            self.current_preview = cached.copy()
            self.preview_display.set_image(self.current_preview)
            return
        self._apply_pipeline_async(
            pipeline=temp_manager,
            description="Updating preview",
            on_finished=self._update_preview_from_result,
        )

    def show_brightness_contrast_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("BrightnessContrast")
        current_alpha = float(step.params.get("alpha", 1.0))
        current_beta = int(step.params.get("beta", 0))
        dlg = BrightnessContrastDialog(alpha=current_alpha, beta=current_beta)

        def _preview(alpha: float, beta: int) -> None:
            self.preview_update("BrightnessContrast", {"alpha": alpha, "beta": beta})

        def _apply() -> None:
            new_alpha, new_beta = dlg.get_values()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("BrightnessContrast", True)
            self.pipeline_manager.update_step_params(
                "BrightnessContrast", {"alpha": new_alpha, "beta": new_beta}
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying brightness/contrast",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "brightness_contrast",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def show_gamma_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("Gamma")
        current_gamma = float(step.params.get("gamma", 1.0))
        dlg = GammaDialog(gamma=current_gamma)

        def _preview(gamma: float) -> None:
            self.preview_update("Gamma", {"gamma": gamma})

        def _apply() -> None:
            new_gamma = dlg.get_value()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("Gamma", True)
            self.pipeline_manager.update_step_params("Gamma", {"gamma": new_gamma})
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying gamma correction",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "gamma",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def show_normalize_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("IntensityNormalization")
        current_alpha = int(step.params.get("alpha", 0))
        current_beta = int(step.params.get("beta", 255))
        dlg = NormalizeDialog(alpha=current_alpha, beta=current_beta)

        def _preview(alpha: int, beta: int) -> None:
            self.preview_update(
                "IntensityNormalization", {"alpha": alpha, "beta": beta}
            )

        def _apply() -> None:
            new_alpha, new_beta = dlg.get_values()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("IntensityNormalization", True)
            self.pipeline_manager.update_step_params(
                "IntensityNormalization", {"alpha": new_alpha, "beta": new_beta}
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying normalization",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "normalize",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def show_noise_reduction_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("NoiseReduction")
        current_method = step.params.get("method", "Gaussian")
        current_ksize = int(step.params.get("ksize", 5))
        dlg = NoiseReductionDialog(method=current_method, ksize=current_ksize)

        def _preview(method: str, ksize: int) -> None:
            self.preview_update("NoiseReduction", {"method": method, "ksize": ksize})

        def _apply() -> None:
            new_method, new_ksize = dlg.get_values()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("NoiseReduction", True)
            self.pipeline_manager.update_step_params(
                "NoiseReduction", {"method": new_method, "ksize": new_ksize}
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying noise reduction",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "noise_reduction",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def show_sharpen_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("Sharpen")
        current_strength = float(step.params.get("strength", 1.0))
        dlg = SharpenDialog(strength=current_strength)

        def _preview(strength: float) -> None:
            self.preview_update("Sharpen", {"strength": strength})

        def _apply() -> None:
            new_strength = dlg.get_value()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("Sharpen", True)
            self.pipeline_manager.update_step_params(
                "Sharpen", {"strength": new_strength}
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying sharpening",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "sharpen",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def show_select_channel_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("SelectChannel")
        current_channel = step.params.get("channel", "All")
        dlg = SelectChannelDialog(current_channel=current_channel)

        def _preview(channel: str) -> None:
            self.preview_update("SelectChannel", {"channel": channel})

        def _apply() -> None:
            new_channel = dlg.get_value()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("SelectChannel", True)
            self.pipeline_manager.update_step_params(
                "SelectChannel", {"channel": new_channel}
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying channel selection",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "select_channel",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def show_crop_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("Crop")
        current_x = int(step.params.get("x_offset", 0))
        current_y = int(step.params.get("y_offset", 0))
        current_width = int(step.params.get("width", 100))
        current_height = int(step.params.get("height", 100))
        dlg = CropDialog(
            x_offset=current_x,
            y_offset=current_y,
            width=current_width,
            height=current_height,
        )

        def _preview(x_offset: int, y_offset: int, width: int, height: int) -> None:
            self.preview_update(
                "Crop",
                {
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "width": width,
                    "height": height,
                },
            )

        def _apply() -> None:
            new_x, new_y, new_width, new_height = dlg.get_values()
            self.pipeline_manager.push_state(
                image=backup, cache_signature=self._committed_signature
            )
            self.pipeline_manager.set_step_enabled("Crop", True)
            self.pipeline_manager.update_step_params(
                "Crop",
                {
                    "x_offset": new_x,
                    "y_offset": new_y,
                    "width": new_width,
                    "height": new_height,
                    "apply_crop": True,
                },
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying crop",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()

        def _cancel() -> None:
            if backup is not None:
                self.preview_display.set_image(backup)
            elif self.base_image is not None:
                self.update_preview()

        self._present_parameter_dialog(
            "crop",
            dlg,
            on_accept=_apply,
            on_cancel=_cancel,
            on_parameters_changed=_preview,
        )

    def showEvent(self, event):  # pragma: no cover - Qt virtual
        super().showEvent(event)
        self.update_undo_redo_actions()


__all__ = [
    "BrightnessContrastDialog",
    "CropDialog",
    "GammaDialog",
    "ImageDisplayWidget",
    "MainWindow",
    "NoiseReductionDialog",
    "NormalizeDialog",
    "SelectChannelDialog",
    "SharpenDialog",
]
