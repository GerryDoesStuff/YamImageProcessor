"""Qt widgets for the preprocessing application."""
from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore, UpdateDispatcher, UpdateMetadata
from core.path_sanitizer import PathValidationError, sanitize_user_path
from core.preprocessing import Config, Loader
from core.thread_controller import OperationCancelled, ThreadController
from plugins.module_base import ModuleBase, ModuleStage
from processing.pipeline_cache import PipelineCacheResult, PipelineCacheTileUpdate
from processing.pipeline_manager import PipelineManager
from processing.preprocessing_pipeline import (
    PreprocessingPipeline,
    build_preprocessing_pipeline,
)
from ui import ModulePane
from ui.control_metadata import ControlMetadata, ControlValueType, get_control_metadata
from ui.theme import (
    SectionWidget,
    ShortcutRegistry,
    ShortcutSummaryWidget,
    ThemedDockWidget,
    load_icon,
    scale_font,
)

from yam_processor.ui import PreviewWidget, TiledImageLevel, TiledImageRecord
from yam_processor.ui.error_reporter import ErrorResolution, present_error_report
from yam_processor.ui.diagnostics_panel import DiagnosticsPanel


LOGGER = logging.getLogger(__name__)


def _is_unified_shell(window: Optional[QtWidgets.QWidget]) -> bool:
    """Return ``True`` when ``window`` hosts the unified multi-stage shell."""

    if window is None:
        return False

    capability = getattr(window, "is_unified_shell", None)
    if callable(capability):
        try:
            return bool(capability())
        except Exception:  # pragma: no cover - defensive probing
            return False

    property_value = window.property("isUnifiedShell")
    if isinstance(property_value, bool):
        return property_value

    return False


@dataclass
class _ProgressivePreviewState:
    """Track incremental pipeline output for the progressive preview."""

    signature: str
    shape: Tuple[int, ...]
    buffer: np.ndarray

    def apply_update(self, update: PipelineCacheTileUpdate) -> None:
        left, top, right, bottom = update.box
        tile = np.asarray(update.tile)
        if tile.dtype != self.buffer.dtype:
            tile = tile.astype(self.buffer.dtype, copy=False)
        if self.buffer.ndim == 2:
            self.buffer[top:bottom, left:right] = tile
        else:
            self.buffer[top:bottom, left:right, ...] = tile


class UpdateNotificationDialog(QtWidgets.QDialog):
    """Simple dialog presenting update notes and helpful links."""

    def __init__(self, metadata: UpdateMetadata, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._metadata = metadata
        self.setWindowTitle(self.tr("Application Update Available"))

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            self.tr("A new version ({version}) is available.").format(version=metadata.version)
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.notes_browser = QtWidgets.QTextBrowser(self)
        self.notes_browser.setReadOnly(True)
        if metadata.notes:
            self.notes_browser.setPlainText(metadata.notes)
        else:
            self.notes_browser.setPlainText(self.tr("No release notes were provided."))
        layout.addWidget(self.notes_browser)

        self.release_notes_link: Optional[QtWidgets.QLabel]
        if metadata.release_notes_url:
            self.release_notes_link = QtWidgets.QLabel(self)
            self.release_notes_link.setText(
                '<a href="{url}">{text}</a>'.format(
                    url=metadata.release_notes_url,
                    text=self.tr("View full release notes"),
                )
            )
            self.release_notes_link.setTextFormat(QtCore.Qt.RichText)
            self.release_notes_link.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
            self.release_notes_link.setOpenExternalLinks(True)
            layout.addWidget(self.release_notes_link)
        else:
            self.release_notes_link = None

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok, parent=self)
        button_box.accepted.connect(self.accept)

        self.download_button: Optional[QtWidgets.QPushButton]
        if metadata.download_url:
            self.download_button = button_box.addButton(
                self.tr("Download"), QtWidgets.QDialogButtonBox.ActionRole
            )
            self.download_button.clicked.connect(self._open_download_url)
        else:
            self.download_button = None

        layout.addWidget(button_box)

    def metadata(self) -> UpdateMetadata:
        return self._metadata

    def _open_download_url(self) -> None:
        if not self._metadata.download_url:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._metadata.download_url))


def _apply_common_metadata(widget: QtWidgets.QWidget, metadata: Optional[ControlMetadata]) -> None:
    if metadata is None:
        return
    tooltip = metadata.tooltip_text()
    if tooltip:
        widget.setToolTip(tooltip)
        widget.setWhatsThis(tooltip)
        widget.setStatusTip(tooltip)


def _configure_spinbox(
    spinbox: QtWidgets.QAbstractSpinBox,
    module_identifier: str,
    parameter_name: str,
    initial_value: Any,
) -> Any:
    metadata = get_control_metadata(module_identifier, parameter_name)
    if metadata is not None:
        if isinstance(spinbox, QtWidgets.QDoubleSpinBox):
            if metadata.minimum is not None:
                spinbox.setMinimum(float(metadata.minimum))
            if metadata.maximum is not None:
                spinbox.setMaximum(float(metadata.maximum))
            if metadata.step is not None:
                spinbox.setSingleStep(float(metadata.step))
            if metadata.decimals is not None:
                spinbox.setDecimals(metadata.decimals)
        elif isinstance(spinbox, QtWidgets.QSpinBox):
            if metadata.minimum is not None:
                spinbox.setMinimum(int(metadata.minimum))
            if metadata.maximum is not None:
                spinbox.setMaximum(int(metadata.maximum))
            if metadata.step is not None:
                spinbox.setSingleStep(int(metadata.step))
        coerced_value = metadata.coerce(initial_value)
        if coerced_value is None:
            coerced_value = metadata.default
        if coerced_value is not None:
            spinbox.setValue(coerced_value)
        _apply_common_metadata(spinbox, metadata)
        return coerced_value

    if initial_value is not None:
        if isinstance(spinbox, QtWidgets.QDoubleSpinBox):
            spinbox.setValue(float(initial_value))
        elif isinstance(spinbox, QtWidgets.QSpinBox):
            spinbox.setValue(int(initial_value))
    return initial_value


def _configure_combobox(
    combobox: QtWidgets.QComboBox,
    module_identifier: str,
    parameter_name: str,
    initial_value: Any,
) -> Any:
    metadata = get_control_metadata(module_identifier, parameter_name)
    if metadata is not None:
        combobox.blockSignals(True)
        try:
            if metadata.choices:
                combobox.clear()
                for option in metadata.choices:
                    combobox.addItem(option.label, option.value)
                    if option.description:
                        combobox.setItemData(
                            combobox.count() - 1, option.description, QtCore.Qt.ToolTipRole
                        )
            value_to_set = metadata.coerce(initial_value)
            if value_to_set is None:
                value_to_set = metadata.default
            if value_to_set is not None:
                index = combobox.findData(value_to_set)
                if index == -1 and isinstance(value_to_set, str):
                    index = combobox.findText(value_to_set)
                if index >= 0:
                    combobox.setCurrentIndex(index)
        finally:
            combobox.blockSignals(False)
        _apply_common_metadata(combobox, metadata)
        current_data = combobox.currentData(QtCore.Qt.UserRole)
        if current_data is not None and metadata.value_type is not ControlValueType.STRING:
            return current_data
        if current_data is not None:
            return current_data
        return combobox.currentText()

    if initial_value is not None:
        combobox.setCurrentText(str(initial_value))
    return combobox.currentText()


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
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        alpha = _configure_spinbox(self.alpha_spin, "BrightnessContrast", "alpha", alpha)
        self.initial_alpha = alpha if alpha is not None else 1.0

        self.beta_spin = QtWidgets.QSpinBox()
        beta = _configure_spinbox(self.beta_spin, "BrightnessContrast", "beta", beta)
        self.initial_beta = beta if beta is not None else 0

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
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        gamma = _configure_spinbox(self.gamma_spin, "Gamma", "gamma", gamma)
        self.initial_gamma = gamma if gamma is not None else 1.0

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
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.alpha_spin = QtWidgets.QSpinBox()
        alpha = _configure_spinbox(self.alpha_spin, "IntensityNormalization", "alpha", alpha)
        self.initial_alpha = alpha if alpha is not None else 0

        self.beta_spin = QtWidgets.QSpinBox()
        beta = _configure_spinbox(self.beta_spin, "IntensityNormalization", "beta", beta)
        self.initial_beta = beta if beta is not None else 255

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
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.method_combo = QtWidgets.QComboBox()
        method = _configure_combobox(self.method_combo, "NoiseReduction", "method", method)
        self.initial_method = method if method is not None else "Gaussian"

        self.ksize_spin = QtWidgets.QSpinBox()
        ksize = _configure_spinbox(self.ksize_spin, "NoiseReduction", "ksize", ksize)
        self.initial_ksize = int(ksize) if ksize is not None else 5

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
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.strength_spin = QtWidgets.QDoubleSpinBox()
        strength = _configure_spinbox(self.strength_spin, "Sharpen", "strength", strength)
        self.initial_strength = strength if strength is not None else 1.0

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
        layout = QtWidgets.QVBoxLayout(self)

        self.channel_combo = QtWidgets.QComboBox()
        current_channel = _configure_combobox(
            self.channel_combo, "SelectChannel", "channel", current_channel
        )
        self.initial_channel = current_channel if current_channel is not None else "All"
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
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.x_spin = QtWidgets.QSpinBox()
        x_offset = _configure_spinbox(self.x_spin, "Crop", "x_offset", x_offset)
        self.initial_x = int(x_offset) if x_offset is not None else 0

        self.y_spin = QtWidgets.QSpinBox()
        y_offset = _configure_spinbox(self.y_spin, "Crop", "y_offset", y_offset)
        self.initial_y = int(y_offset) if y_offset is not None else 0

        self.width_spin = QtWidgets.QSpinBox()
        width = _configure_spinbox(self.width_spin, "Crop", "width", width)
        self.initial_width = int(width) if width is not None else 100

        self.height_spin = QtWidgets.QSpinBox()
        height = _configure_spinbox(self.height_spin, "Crop", "height", height)
        self.initial_height = int(height) if height is not None else 100

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


class PreprocessingPane(ModulePane):
    pipelineDockVisibilityChanged = QtCore.pyqtSignal(bool)
    moduleControlsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsLoggingToggled = QtCore.pyqtSignal(bool)
    moduleActivated = QtCore.pyqtSignal(str)

    def __init__(
        self,
        app_core: AppCore,
        *,
        host: QtWidgets.QMainWindow,
    ):
        super().__init__(host)
        self._window = host
        unified_shell = _is_unified_shell(self._window)
        self.app_core = app_core
        if not unified_shell:
            self._window.setWindowTitle("Image Pre-Processing Module")
            self._window.resize(1200, 700)
            window_icon = load_icon(
                "manage_modules",
                fallback=self._window.style().standardIcon(
                    QtWidgets.QStyle.SP_DesktopIcon
                ),
            )
            if not window_icon.isNull():
                self._window.setWindowIcon(window_icon)
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
        self._pending_preview_signature: Optional[str] = None
        self._progressive_preview_state: Optional[_ProgressivePreviewState] = None
        self._progressive_generation_counter = 0
        self._active_progressive_generation: Optional[int] = None
        self._progressive_previous_frame: Optional[np.ndarray] = None
        if self.app_core.thread_controller is None:
            self.app_core.thread_controller = ThreadController(parent=self._window)
        self.thread_controller: ThreadController = self.app_core.thread_controller
        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._register_thread_signals()
        # Track floating parameter editors so they can eventually be migrated into
        # docked widgets that share the asynchronous preview stream (see Issue #1).
        self._active_parameter_dialogs: Dict[str, QtWidgets.QDialog] = {}
        self._parameter_stream_sources: Dict[str, Any] = {}

        self.image_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.image_splitter.setObjectName("imageDisplaySplitter")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_splitter, 1)

        original_section = SectionWidget("Original Image", self.image_splitter)
        original_section.setObjectName("originalImageSection")
        original_section.setAccessibleName("Original image panel")
        original_layout = original_section.layout
        self.original_display = PreviewWidget(original_section)
        self.original_display.setObjectName("originalImageDisplay")
        self.original_display.setFocusPolicy(QtCore.Qt.StrongFocus)
        original_layout.addWidget(self.original_display, 1)

        preview_section = SectionWidget("Pre-Processing Preview", self.image_splitter)
        preview_section.setObjectName("previewImageSection")
        preview_section.setAccessibleName("Pre-processing preview panel")
        preview_layout = preview_section.layout
        self.preview_display = PreviewWidget(preview_section)
        self.preview_display.setObjectName("previewImageDisplay")
        self.preview_display.setFocusPolicy(QtCore.Qt.StrongFocus)
        preview_layout.addWidget(self.preview_display, 1)

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

        self.pipeline_dock = ThemedDockWidget("Pipeline Overview", self._window)
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
        self._window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.pipeline_dock)
        self.pipeline_dock.visibilityChanged.connect(
            self._on_pipeline_dock_visibility_changed
        )

        diagnostics_widget = SectionWidget("Diagnostics & Shortcuts")
        diagnostics_widget.setObjectName("diagnosticsSection")
        diagnostics_layout = diagnostics_widget.layout
        self.shortcut_summary = ShortcutSummaryWidget()
        diagnostics_layout.addWidget(self.shortcut_summary)
        self.diagnostics_panel = DiagnosticsPanel()
        self.diagnostics_panel.setObjectName("diagnosticsPanel")
        self.diagnostics_panel.setAccessibleName("Diagnostics activity viewer")
        self.diagnostics_panel.setFocusPolicy(QtCore.Qt.StrongFocus)
        diagnostics_layout.addWidget(self.diagnostics_panel, 1)

        self.diagnostics_dock = ThemedDockWidget("Diagnostics Log", self._window)
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
        self._window.addDockWidget(
            QtCore.Qt.BottomDockWidgetArea, self.diagnostics_dock
        )
        self.diagnostics_dock.visibilityChanged.connect(
            self._on_diagnostics_dock_visibility_changed
        )

        panel_handler = self.diagnostics_panel.log_handler()
        if self.app_core.log_handler is not None and getattr(
            self.app_core.log_handler, "formatter", None
        ) is not None:
            panel_handler.setFormatter(self.app_core.log_handler.formatter)
        else:
            panel_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
        self.diagnostics_panel.attach_to_logger(logging.getLogger())
        self.diagnostics_panel.set_thread_controller(self.thread_controller)
        self._task_counter = 0
        self._active_task_id: Optional[str] = None
        self._module_task_ids: Dict[str, str] = {}
        self._register_module_health_entries()
        self.moduleActivated.connect(self._on_module_activated)

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
            "Module Parameters", self._window
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
        self._window.addDockWidget(
            QtCore.Qt.LeftDockWidgetArea, self.module_controls_dock
        )
        self.module_controls_dock.visibilityChanged.connect(
            self._on_module_controls_dock_visibility_changed
        )

        self.shortcut_status_label = QtWidgets.QLabel()
        self.shortcut_status_label.setObjectName("shortcutStatusLabel")
        self.shortcut_status_label.setAccessibleName(
            "Primary keyboard shortcuts summary"
        )
        self._window.statusBar().addPermanentWidget(self.shortcut_status_label, 1)
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
            self._focus_diagnostics_panel
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

        self._show_status_message("Ready")

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

    def _status_bar(self) -> QtWidgets.QStatusBar:
        return self._window.statusBar()

    def _show_status_message(self, message: str, timeout: int = 0) -> None:
        status_bar = self._status_bar()
        if status_bar is not None:
            status_bar.showMessage(message, timeout)

    def _build_preview_record(
        self, image: Optional[np.ndarray]
    ) -> Optional[TiledImageRecord]:
        if image is None:
            return None
        array = np.ascontiguousarray(image)
        if array.ndim < 2:
            return None
        levels: List[Tuple[float, np.ndarray]] = [(1.0, array)]
        current = array
        scale = 1.0
        while min(current.shape[0], current.shape[1]) > 512:
            new_width = max(1, current.shape[1] // 2)
            new_height = max(1, current.shape[0] // 2)
            resized = cv2.resize(
                current,
                (int(new_width), int(new_height)),
                interpolation=cv2.INTER_AREA,
            )
            scale /= 2.0
            current = np.ascontiguousarray(resized)
            levels.append((scale, current))
        tiled_levels = [
            TiledImageLevel(scale=level_scale, fetch=lambda data=data: data)
            for level_scale, data in levels
        ]
        return TiledImageRecord(tiled_levels)

    def _set_original_display_image(self, image: Optional[np.ndarray]) -> None:
        self.original_display.set_image(self._build_preview_record(image))

    def _set_preview_display_image(self, image: Optional[np.ndarray]) -> None:
        self.preview_display.set_image(self._build_preview_record(image))

    def _update_preview_display_buffer(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self.preview_display.set_image(None)
            return
        self.preview_display.update_array(np.asarray(image))

    def _current_preview_array(self) -> Optional[np.ndarray]:
        if self.current_preview is not None:
            return np.array(self.current_preview, copy=True)
        return self.preview_display.current_array()

    def _restore_progressive_baseline(self) -> None:
        if self._progressive_previous_frame is None:
            return
        baseline = np.array(self._progressive_previous_frame, copy=True)
        self._progressive_previous_frame = None
        self.preview_display.update_array(baseline)

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
            self._update_preview_display_buffer(self.current_preview)
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
            self._update_preview_display_buffer(self.current_preview)
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

    # ------------------------------------------------------------------
    # ModulePane implementation
    def on_activated(self) -> None:
        self.update_undo_redo_actions()

    def on_deactivated(self) -> None:
        pass

    def save_outputs(self) -> None:
        self.save_processed_image()

    def update_pipeline_summary(self) -> None:
        self.update_pipeline_label()

    def set_diagnostics_visible(self, visible: bool) -> None:
        self.diagnostics_dock.setVisible(visible)

    def teardown(self) -> None:
        if hasattr(self, "diagnostics_panel") and self.diagnostics_panel is not None:
            self.diagnostics_panel.detach_from_logger()
        for dialog in list(self._active_parameter_dialogs.values()):
            dialog.close()
        self._active_parameter_dialogs.clear()
        self._parameter_stream_sources.clear()

    def build_menu(self):
        menubar = self._window.menuBar()
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

        self.load_image_action = QtWidgets.QAction("&Load Image...", self._window)
        self.load_image_action.setShortcut(QtGui.QKeySequence.Open)
        self.load_image_action.setShortcutVisibleInContextMenu(True)
        self.load_image_action.setStatusTip("Load an image for preprocessing")
        self.load_image_action.triggered.connect(self.load_image)
        self.load_image_action.setIcon(
            load_icon(
                "open_project",
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton),
            )
        )
        file_menu.addAction(self.load_image_action)
        self.shortcut_registry.register_action("Load image", self.load_image_action)

        self.save_processed_image_action = QtWidgets.QAction(
            "&Save Pre-Processed Image...", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            )
        )
        file_menu.addAction(self.save_processed_image_action)
        self.shortcut_registry.register_action(
            "Save pre-processed image", self.save_processed_image_action
        )

        file_menu.addSeparator()

        self.mass_preprocess_action = QtWidgets.QAction(
            "Mass Pre-Process &Folder...", self._window
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
                fallback=self._window.style().standardIcon(
                    QtWidgets.QStyle.SP_FileDialogDetailedView
                ),
            )
        )
        file_menu.addAction(self.mass_preprocess_action)
        self.shortcut_registry.register_action(
            "Mass pre-process folder", self.mass_preprocess_action
        )

        file_menu.addSeparator()

        self.import_pipeline_action = QtWidgets.QAction(
            "&Import Pipeline Settings...", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton),
            )
        )
        file_menu.addAction(self.import_pipeline_action)
        self.shortcut_registry.register_action(
            "Import pipeline settings", self.import_pipeline_action
        )

        self.export_pipeline_action = QtWidgets.QAction(
            "E&xport Pipeline Settings...", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            )
        )
        file_menu.addAction(self.export_pipeline_action)
        self.shortcut_registry.register_action(
            "Export pipeline settings", self.export_pipeline_action
        )

        edit_menu = ensure_menu(("Edit",))
        self.undo_action = QtWidgets.QAction("&Undo", self._window)
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)
        self.undo_action.setShortcutVisibleInContextMenu(True)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        self.undo_action.setIcon(
            load_icon(
                "undo",
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack),
            )
        )
        edit_menu.addAction(self.undo_action)
        self.shortcut_registry.register_action("Undo", self.undo_action)

        self.redo_action = QtWidgets.QAction("&Redo", self._window)
        self.redo_action.setShortcut(QtGui.QKeySequence.Redo)
        self.redo_action.setShortcutVisibleInContextMenu(True)
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        self.redo_action.setIcon(
            load_icon(
                "redo",
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward),
            )
        )
        edit_menu.addAction(self.redo_action)
        self.shortcut_registry.register_action("Redo", self.redo_action)

        edit_menu.addSeparator()

        self.reset_pipeline_action = QtWidgets.QAction("Reset &All", self._window)
        self.reset_pipeline_action.setShortcut("Ctrl+Shift+R")
        self.reset_pipeline_action.setShortcutVisibleInContextMenu(True)
        self.reset_pipeline_action.setStatusTip(
            "Restore the preprocessing pipeline to default settings"
        )
        self.reset_pipeline_action.triggered.connect(self.reset_all)
        self.reset_pipeline_action.setIcon(
            load_icon(
                "redo",
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload),
            )
        )
        edit_menu.addAction(self.reset_pipeline_action)
        self.shortcut_registry.register_action(
            "Reset pipeline", self.reset_pipeline_action
        )

        view_menu = ensure_menu(("View",))

        self.show_pipeline_dock_action = QtWidgets.QAction(
            "Pipeline Overview", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView),
            )
        )
        view_menu.addAction(self.show_pipeline_dock_action)
        self.shortcut_registry.register_action(
            "Toggle pipeline overview", self.show_pipeline_dock_action
        )

        self.show_diagnostics_dock_action = QtWidgets.QAction(
            "Diagnostics Log", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_FileDialogInfoView),
            )
        )
        view_menu.addAction(self.show_diagnostics_dock_action)
        self.shortcut_registry.register_action(
            "Toggle diagnostics log", self.show_diagnostics_dock_action
        )

        self.show_module_controls_dock_action = QtWidgets.QAction(
            "Module Controls", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
            )
        )
        view_menu.addAction(self.show_module_controls_dock_action)
        self.shortcut_registry.register_action(
            "Toggle module controls", self.show_module_controls_dock_action
        )

        view_menu.addSeparator()

        self.enable_diagnostics_logging_action = QtWidgets.QAction(
            "Enable Diagnostics Logging", self._window
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

        self.enable_telemetry_action = QtWidgets.QAction(
            "Share Anonymous Telemetry", self._window
        )
        self.enable_telemetry_action.setCheckable(True)
        self.enable_telemetry_action.setChecked(self.app_core.telemetry_opt_in)
        self.enable_telemetry_action.setEnabled(self.app_core.diagnostics_enabled)
        self.enable_telemetry_action.setStatusTip(
            "Allow the application to send anonymized usage diagnostics"
        )
        self.enable_telemetry_action.toggled.connect(self._set_telemetry_opt_in)
        view_menu.addAction(self.enable_telemetry_action)

        ensure_menu(("Modules",))

        for module in self.app_core.get_modules(ModuleStage.PREPROCESSING):
            for entry in module.menu_entries():
                path = entry.path
                if not path or path[0] != "Modules":
                    path = ("Modules", *path)
                menu = ensure_menu(path)
                action = QtWidgets.QAction(entry.text, self._window)
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
            "View &Documentation", self._window
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
                fallback=self._window.style().standardIcon(QtWidgets.QStyle.SP_DialogHelpButton),
            )
        )
        help_menu.addAction(self.view_documentation_action)
        self.shortcut_registry.register_action(
            "View documentation", self.view_documentation_action
        )

        self.about_action = QtWidgets.QAction("&About", self._window)
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
                fallback=self._window.style().standardIcon(
                    QtWidgets.QStyle.SP_FileDialogInfoView
                ),
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
            ("Focus diagnostics panel", "_diagnostics_focus_shortcut"),
            ("Focus module controls", "_module_controls_focus_shortcut"),
            ("Undo", "undo_shortcut"),
            ("Redo", "redo_shortcut"),
        )
        for label, attr in mapping:
            shortcut = getattr(self, attr, None)
            if shortcut is not None:
                self.shortcut_registry.register_shortcut(label, shortcut)

    def _report_error(
        self,
        message: str,
        *,
        context: Optional[Dict[str, object]] = None,
        window_title: Optional[str] = None,
        enable_retry: bool = False,
        retry_label: Optional[str] = None,
        retry_callback: Optional[Callable[[], None]] = None,
        fallback_traceback: Optional[str] = None,
    ) -> None:
        metadata: Dict[str, object] = {"module": "preprocessing"}
        if context:
            metadata.update(context)
        recovery_summary = None
        recovery_status_message: Optional[str] = None
        recovery_status_is_error = False
        enable_discard_autosave = False
        recovery_manager = getattr(self.app_core, "recovery_manager", None)
        if recovery_manager is not None:
            try:
                recovery_summary = recovery_manager.summary()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.debug(
                    "Failed to summarise recovery state",
                    extra={"component": "preprocessing", "error": str(exc)},
                )
            else:
                metadata.setdefault("recovery", recovery_summary.to_metadata())
                status = recovery_summary.status_message()
                if status is not None:
                    recovery_status_message, recovery_status_is_error = status
                enable_discard_autosave = recovery_summary.has_snapshot
        resolution = present_error_report(
            message,
            logger=LOGGER,
            parent=self,
            window_title=window_title,
            metadata=metadata,
            enable_retry=enable_retry and retry_callback is not None,
            retry_label=retry_label,
            fallback_traceback=fallback_traceback,
            enable_discard=enable_discard_autosave,
            discard_label=self.tr("&Discard autosave"),
            status_message=recovery_status_message,
            status_is_error=recovery_status_is_error,
        )
        if (
            resolution is ErrorResolution.DISCARD_AUTOSAVE
            and recovery_manager is not None
            and recovery_summary is not None
            and recovery_summary.has_snapshot
        ):
            recovery_manager.discard_pending_snapshot()
        if resolution is ErrorResolution.RETRY and retry_callback is not None:
            try:
                retry_callback()
            except Exception as retry_error:  # pragma: no cover - user facing retry handling
                retry_context = dict(context or {})
                retry_context["retry_failed"] = True
                retry_context["retry_error"] = str(retry_error)
                self._report_error(
                    self.tr("Retry failed: {error}").format(error=retry_error),
                    context=retry_context,
                    window_title=window_title,
                    enable_retry=enable_retry,
                    retry_label=retry_label,
                    retry_callback=retry_callback,
                    fallback_traceback=traceback.format_exc(),
                )

    def _sanitize_dialog_path(
        self,
        raw_path: str,
        *,
        allow_directory: bool,
        allow_file: bool,
        must_exist: bool,
        operation: str,
        window_title: str,
        message_template: str,
        extra_context: Optional[Dict[str, object]] = None,
    ) -> Optional[Path]:
        if not raw_path:
            return None
        try:
            return sanitize_user_path(
                raw_path,
                must_exist=must_exist,
                allow_directory=allow_directory,
                allow_file=allow_file,
            )
        except PathValidationError as exc:
            context: Dict[str, object] = {"operation": operation, "path": raw_path}
            if extra_context:
                context.update(extra_context)
            self._report_error(
                message_template.format(error=exc),
                context=context,
                window_title=window_title,
            )
            return None

    def _activate_module(self, module: ModuleBase) -> None:
        identifier = module.metadata.identifier
        self._update_module_status(identifier, self.tr("Activating"), progress=0.0)
        try:
            module.activate(self)
            LOGGER.info("Module %s activated", identifier)
            self.moduleActivated.emit(identifier)
        except NotImplementedError:
            self._update_module_status(identifier, self.tr("Unavailable"), progress=0.0)
            logging.warning(
                "Module %s does not implement an activation handler",
                identifier,
            )
            self._show_status_message(
                f"{module.metadata.title} is not available for activation.", 2000
            )
        except Exception as exc:  # pragma: no cover - defensive UI guard
            self._update_module_status(identifier, self.tr("Error"), progress=0.0)
            context = {
                "operation": "activate_module",
                "module": identifier,
                "title": module.metadata.title,
            }
            self._report_error(
                self.tr("{title} failed to run.\n{error}").format(
                    title=module.metadata.title,
                    error=exc,
                ),
                context=context,
                window_title=self.tr("Module Error"),
                enable_retry=True,
                retry_label=self.tr("&Retry Activation"),
                retry_callback=lambda: self._activate_module(module),
            )
            self._show_status_message("Error running module action", 4000)

    def _register_thread_signals(self) -> None:
        self._current_task_description: str = ""
        self.thread_controller.task_started.connect(self._on_task_started)
        self.thread_controller.task_progress.connect(self._on_task_progress)
        self.thread_controller.task_finished.connect(self._on_task_finished)
        self.thread_controller.task_canceled.connect(self._on_task_canceled)
        self.thread_controller.task_failed.connect(self._on_task_failed)

    def diagnostics_panel_widget(self) -> Optional[DiagnosticsPanel]:
        return getattr(self, "diagnostics_panel", None)

    def _register_module_health_entries(self) -> None:
        if not hasattr(self, "diagnostics_panel") or self.diagnostics_panel is None:
            return
        self._module_task_ids.clear()
        for module in self.app_core.get_modules(ModuleStage.PREPROCESSING):
            identifier = module.metadata.identifier
            task_id = f"module::{identifier}"
            self._module_task_ids[identifier] = task_id
            self.diagnostics_panel.register_task(task_id, module.metadata.title)
            self._update_module_status(identifier, self.tr("Ready"), progress=1.0)

    def _update_module_status(
        self, identifier: str, status: str, *, progress: Optional[float] = None
    ) -> None:
        if not hasattr(self, "diagnostics_panel") or self.diagnostics_panel is None:
            return
        task_id = self._module_task_ids.get(identifier)
        if task_id is None:
            return
        self.diagnostics_panel.update_task_status(task_id, status)
        if progress is not None:
            self.diagnostics_panel.update_task_progress(task_id, progress)

    @QtCore.pyqtSlot(str)
    def _on_module_activated(self, identifier: str) -> None:
        self._update_module_status(identifier, self.tr("Activated"), progress=1.0)

        def _reset_status() -> None:
            self._update_module_status(identifier, self.tr("Ready"), progress=1.0)

        QtCore.QTimer.singleShot(2000, _reset_status)

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

    def _focus_diagnostics_panel(self) -> None:
        if not hasattr(self, "diagnostics_panel") or self.diagnostics_panel is None:
            return
        self.diagnostics_dock.setVisible(True)
        self.diagnostics_panel.focus_logs()

    def _set_diagnostics_logging(self, enabled: bool) -> None:
        self.app_core.set_diagnostics_enabled(enabled)
        if hasattr(self, "enable_telemetry_action"):
            self.enable_telemetry_action.setEnabled(enabled)
            self.enable_telemetry_action.blockSignals(True)
            self.enable_telemetry_action.setChecked(self.app_core.telemetry_opt_in)
            self.enable_telemetry_action.blockSignals(False)
        message = (
            "Diagnostics logging enabled."
            if enabled
            else "Diagnostics logging disabled."
        )
        LOGGER.info(message)
        self._show_status_message(message, 3000)
        self.diagnosticsLoggingToggled.emit(enabled)

    def _set_telemetry_opt_in(self, opt_in: bool) -> None:
        self.app_core.configure_telemetry(opt_in)
        if hasattr(self, "enable_telemetry_action"):
            self.enable_telemetry_action.blockSignals(True)
            self.enable_telemetry_action.setChecked(self.app_core.telemetry_opt_in)
            self.enable_telemetry_action.setEnabled(self.app_core.diagnostics_enabled)
            self.enable_telemetry_action.blockSignals(False)

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
                self._show_status_message(
                    f"Opened documentation: {candidate.name}", 3000
                )
                return
        QtWidgets.QMessageBox.information(
            self,
            "Documentation",
            "Documentation files are not available in this build.",
        )

    @QtCore.pyqtSlot(str)
    def _on_task_started(self, description: str) -> None:
        self._current_task_description = description or "Processing"
        self._show_status_message(self._current_task_description)
        self._task_counter += 1
        task_id = f"task::{self._task_counter}"
        self._active_task_id = task_id
        if self.diagnostics_panel is not None:
            self.diagnostics_panel.register_task(
                task_id, self._current_task_description
            )
            self.diagnostics_panel.update_task_status(
                task_id, self.tr("Running")
            )
        LOGGER.info("%s started.", self._current_task_description)

    @QtCore.pyqtSlot(int)
    def _on_task_progress(self, value: int) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.setValue(value)
        if self.diagnostics_panel is not None and self._active_task_id is not None:
            self.diagnostics_panel.update_task_progress(
                self._active_task_id, value / 100.0
            )

    @QtCore.pyqtSlot(object)
    def _on_task_finished(self, _result: object) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        self._show_status_message("Ready", 1500)
        if self.diagnostics_panel is not None and self._active_task_id is not None:
            self.diagnostics_panel.complete_task(self._active_task_id)
        LOGGER.info(
            "%s completed successfully.",
            self._current_task_description or "Processing",
        )
        self._active_task_id = None

    @QtCore.pyqtSlot()
    def _on_task_canceled(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        self._show_status_message("Operation canceled", 2000)
        if self.diagnostics_panel is not None and self._active_task_id is not None:
            self.diagnostics_panel.update_task_status(
                self._active_task_id, self.tr("Canceled")
            )
        LOGGER.warning(
            "%s was canceled.",
            self._current_task_description or "Processing",
        )
        self._active_progressive_generation = None
        self._progressive_preview_state = None
        self._pending_preview_signature = None
        self._restore_progressive_baseline()
        self._active_task_id = None

    @QtCore.pyqtSlot(Exception, str)
    def _on_task_failed(self, error: Exception, stack: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        if self.diagnostics_panel is not None and self._active_task_id is not None:
            self.diagnostics_panel.update_task_status(
                self._active_task_id,
                self.tr("Failed"),
            )
        context: Dict[str, object] = {
            "operation": "pipeline_task",
            "task": self._current_task_description or "Processing",
        }
        if stack:
            context["stack"] = stack
        self._report_error(
            self.tr("{task} failed: {error}").format(
                task=self._current_task_description or self.tr("Processing"),
                error=error,
            ),
            context=context,
            window_title=self.tr("Processing Error"),
            fallback_traceback=stack,
        )
        self._show_status_message("Error during processing", 4000)
        LOGGER.error(
            "%s failed: %s",
            self._current_task_description or "Processing",
            error,
        )
        self._active_progressive_generation = None
        self._progressive_preview_state = None
        self._pending_preview_signature = None
        self._restore_progressive_baseline()
        self._active_task_id = None

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
        final_signature, _ = self.pipeline_cache.predict(source_id, steps)

        self._pending_preview_signature = final_signature
        self._progressive_preview_state = None
        self._progressive_generation_counter += 1
        generation = self._progressive_generation_counter
        self._active_progressive_generation = generation
        self._progressive_previous_frame = self._current_preview_array()

        def _handle_finished(result: PipelineCacheResult) -> None:
            if self._active_progressive_generation != generation:
                return
            self._active_progressive_generation = None
            self._progressive_preview_state = None
            self._pending_preview_signature = None
            self._progressive_previous_frame = None
            if on_finished is not None:
                on_finished(result)

        def _handle_canceled() -> None:
            if self._active_progressive_generation == generation:
                self._active_progressive_generation = None
                self._progressive_preview_state = None
                self._pending_preview_signature = None
                self._restore_progressive_baseline()
            if on_canceled is not None:
                on_canceled()

        def _handle_incremental(update: object) -> None:
            self._handle_pipeline_incremental_update(update, generation)

        def _task(
            cancel_event: threading.Event,
            progress: Callable[[int], None],
            incremental: Optional[Callable[[object], None]] = None,
        ):
            return self.pipeline_cache.compute(
                source_id=source_id,
                image=image,
                steps=steps,
                cancel_event=cancel_event,
                progress=progress,
                incremental=incremental,
            )

        self.thread_controller.run_task(
            _task,
            description=description,
            on_finished=_handle_finished,
            on_canceled=_handle_canceled,
            on_intermediate=_handle_incremental,
        )

    def _handle_pipeline_incremental_update(
        self, update: object, generation: int
    ) -> None:
        if self._active_progressive_generation != generation:
            return
        if not isinstance(update, PipelineCacheTileUpdate):
            return
        if (
            self._pending_preview_signature is not None
            and update.final_signature != self._pending_preview_signature
        ):
            return
        if update.step_index != update.total_steps:
            return

        state = self._progressive_preview_state
        shape = tuple(int(dim) for dim in update.shape)
        if (
            state is None
            or state.signature != update.final_signature
            or state.shape != shape
        ):
            baseline = self._progressive_previous_frame
            if baseline is not None and baseline.shape == shape:
                if baseline.dtype != update.dtype:
                    buffer = baseline.astype(update.dtype, copy=True)
                else:
                    buffer = np.array(baseline, copy=True)
            else:
                buffer = np.zeros(shape, dtype=update.dtype)
            state = _ProgressivePreviewState(
                signature=update.final_signature,
                shape=shape,
                buffer=buffer,
            )
            self._progressive_preview_state = state

        state.apply_update(update)
        self.preview_display.update_array(state.buffer)

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
        self._update_preview_display_buffer(self.current_preview)

    def _update_committed_from_result(self, result: PipelineCacheResult) -> None:
        self.committed_image = result.image.copy()
        self.current_preview = self.committed_image.copy()
        self._update_preview_display_buffer(self.current_preview)
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
        self._show_status_message("Reset all processing to defaults.", 3000)

    def mass_preprocess(self):
        raw_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder for Mass Pre-Process"
        )
        folder_path = self._sanitize_dialog_path(
            raw_folder,
            allow_directory=True,
            allow_file=False,
            must_exist=True,
            operation="mass_preprocess",
            window_title=self.tr("Mass Pre-Process"),
            message_template=self.tr("The selected folder could not be used: {error}"),
            extra_context={"dialog": "mass_preprocess"},
        )
        if folder_path is None:
            return

        output_folder = folder_path.parent / f"{folder_path.name}_pp"
        output_folder.mkdir(parents=True, exist_ok=True)
        files = [
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in Config.SUPPORTED_FORMATS
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
                filename = path.name
                try:
                    image = Loader.load_image(str(path))
                    result = pipeline_snapshot.apply(image)
                    new_filename = f"{path.stem}_pp{path.suffix}"
                    outpath = output_folder / new_filename
                    io_manager.save_image(
                        outpath,
                        result,
                        metadata={
                            "stage": "preprocessing",
                            "mode": "batch",
                            "source": {
                                "input": str(path),
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
        destination = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=False,
            operation="export_pipeline",
            window_title=self.tr("Pipeline Export"),
            message_template=self.tr("The selected file path could not be used: {error}"),
        )
        if destination is None:
            return

        def _attempt_export() -> None:
            payload = json.dumps(self.pipeline_manager.to_dict(), indent=2)
            destination.write_text(payload, encoding="utf-8")
            QtWidgets.QMessageBox.information(
                self, "Pipeline Export", "Pipeline settings exported."
            )

        try:
            _attempt_export()
        except Exception as exc:  # pragma: no cover - user feedback
            self._report_error(
                self.tr("Failed to export pipeline: {error}").format(error=exc),
                context={
                    "operation": "export_pipeline",
                    "destination": str(destination),
                },
                window_title=self.tr("Pipeline Export"),
                enable_retry=True,
                retry_label=self.tr("&Retry Export"),
                retry_callback=_attempt_export,
                fallback_traceback=traceback.format_exc(),
            )

    def import_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Pipeline Settings", "", "JSON Files (*.json)"
        )
        source = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=True,
            operation="import_pipeline",
            window_title=self.tr("Pipeline Import"),
            message_template=self.tr("The selected file could not be used: {error}"),
        )
        if source is None:
            return

        def _attempt_import() -> None:
            payload = source.read_text(encoding="utf-8")
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

        try:
            _attempt_import()
        except Exception as exc:  # pragma: no cover - user feedback
            self._report_error(
                self.tr("Failed to import pipeline: {error}").format(error=exc),
                context={
                    "operation": "import_pipeline",
                    "source": str(source),
                },
                window_title=self.tr("Pipeline Import"),
                enable_retry=True,
                retry_label=self.tr("&Retry Import"),
                retry_callback=_attempt_import,
                fallback_traceback=traceback.format_exc(),
            )

    def load_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.jpg *.png *.tiff *.bmp *.npy)"
        )
        path = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=True,
            operation="load_image",
            window_title=self.tr("Image Load Error"),
            message_template=self.tr("The selected image could not be used: {error}"),
        )
        if path is None:
            return

        def _attempt_load() -> None:
            image = Loader.load_image(str(path))
            if self._source_id is not None:
                self.pipeline_cache.discard_cache(self._source_id)
            self.original_image = image.copy()
            self.base_image = image.copy()
            self.committed_image = image.copy()
            self.current_preview = image.copy()
            self.current_image_path = str(path)
            self._source_id = self.pipeline_cache.register_source(
                self.base_image, hint=str(path)
            )
            self._committed_signature = self._source_id
            self._preview_signature = self._source_id
            self._last_pipeline_metadata = self.pipeline_cache.metadata_for(
                self._source_id, self._source_id
            )
            self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
            self._set_original_display_image(self.original_image)
            self._set_preview_display_image(self.base_image)
            self.update_preview()
            self.pipeline_manager.clear_history()
            self.update_undo_redo_actions()
            self._show_status_message(f"Loaded image: {path}")

        try:
            _attempt_load()
        except Exception as exc:  # pragma: no cover - user feedback
            self._report_error(
                self.tr("Failed to load image: {error}").format(error=exc),
                context={
                    "operation": "load_image",
                    "source": str(path),
                },
                window_title=self.tr("Image Load Error"),
                enable_retry=True,
                retry_label=self.tr("&Retry Load"),
                retry_callback=_attempt_load,
                fallback_traceback=traceback.format_exc(),
            )
            return

    def save_processed_image(self):
        if self.committed_image is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "No processed image to save.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Pre-Processed Image", "", "Image Files (*.jpg *.png *.tiff *.bmp *.npy)"
        )
        destination = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=False,
            operation="save_processed_image",
            window_title=self.tr("Save Pre-Processed Image"),
            message_template=self.tr("The selected file path could not be used: {error}"),
        )
        if destination is None:
            return
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
            destination,
            self.committed_image,
            metadata=metadata,
            pipeline=pipeline_metadata,
            settings_snapshot=settings_snapshot,
        )
        self._show_status_message(
            f"Saved processed image to: {result.image_path}"
        )

    def update_preview(self):
        if self.base_image is None or self._source_id is None or self.pipeline is None:
            return
        steps = tuple(step.clone() for step in self.pipeline.steps)
        final_signature, _ = self.pipeline_cache.predict(self._source_id, steps)
        cached = self.pipeline_cache.get_cached_image(self._source_id, final_signature)
        if cached is not None:
            self._preview_signature = final_signature
            self.current_preview = cached.copy()
            self._update_preview_display_buffer(self.current_preview)
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
            self._update_preview_display_buffer(self.current_preview)
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
                self._update_preview_display_buffer(backup)
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
                self._update_preview_display_buffer(backup)
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
                self._update_preview_display_buffer(backup)
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
                self._update_preview_display_buffer(backup)
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
                self._set_preview_display_image(backup)
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
                self._set_preview_display_image(backup)
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
                self._set_preview_display_image(backup)
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



class ModuleWindow(QtWidgets.QMainWindow):
    pipelineDockVisibilityChanged = QtCore.pyqtSignal(bool)
    moduleControlsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsDockVisibilityChanged = QtCore.pyqtSignal(bool)
    diagnosticsLoggingToggled = QtCore.pyqtSignal(bool)
    moduleActivated = QtCore.pyqtSignal(str)

    def __init__(self, app_core: AppCore):
        super().__init__()
        self.setObjectName("preprocessingMainWindow")
        self.app_core = app_core
        self._init_update_notifications()
        self.pane = PreprocessingPane(app_core, host=self)
        self.setCentralWidget(self.pane)
        self._forward_signals()

    # ------------------------------------------------------------------
    # Update notification helpers
    def _init_update_notifications(self) -> None:
        self._pending_update: Optional[UpdateMetadata] = None
        self._update_dispatcher: Optional[UpdateDispatcher] = None
        self._update_dialog_factory: Callable[[UpdateMetadata], QtWidgets.QDialog] = (
            lambda metadata: UpdateNotificationDialog(metadata, self)
        )
        dispatcher = getattr(self.app_core, "update_dispatcher", None)
        if isinstance(dispatcher, UpdateDispatcher):
            self.set_update_dispatcher(dispatcher)

    def set_update_dispatcher(self, dispatcher: UpdateDispatcher) -> None:
        if self._update_dispatcher is dispatcher:
            return
        if self._update_dispatcher is not None:
            self._update_dispatcher.remove_listener(self._on_update_available)
        self._update_dispatcher = dispatcher
        dispatcher.add_listener(self._on_update_available)

    def _on_update_available(self, metadata: UpdateMetadata) -> None:
        self._pending_update = metadata
        dialog = self._create_update_dialog(metadata)
        try:
            dialog.exec_()
        finally:
            self.acknowledge_available_update()
            self._pending_update = None

    def _create_update_dialog(self, metadata: UpdateMetadata) -> QtWidgets.QDialog:
        return self._update_dialog_factory(metadata)

    def acknowledge_available_update(self) -> None:
        if self._update_dispatcher is not None:
            self._update_dispatcher.acknowledge()

    # ------------------------------------------------------------------
    # QWidget overrides
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._update_dispatcher is not None:
            self._update_dispatcher.remove_listener(self._on_update_available)
        self.pane.teardown()
        super().closeEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.pane.update_undo_redo_actions()

    # ------------------------------------------------------------------
    # Delegation helpers
    def _forward_signals(self) -> None:
        self.pane.pipelineDockVisibilityChanged.connect(
            self.pipelineDockVisibilityChanged
        )
        self.pane.moduleControlsDockVisibilityChanged.connect(
            self.moduleControlsDockVisibilityChanged
        )
        self.pane.diagnosticsDockVisibilityChanged.connect(
            self.diagnosticsDockVisibilityChanged
        )
        self.pane.diagnosticsLoggingToggled.connect(self.diagnosticsLoggingToggled)
        self.pane.moduleActivated.connect(self.moduleActivated)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.pane, item)

    def on_activated(self) -> None:
        self.pane.on_activated()

    def on_deactivated(self) -> None:
        self.pane.on_deactivated()

    def load_image(self) -> None:
        self.pane.load_image()

    def save_outputs(self) -> None:
        self.pane.save_outputs()

    def update_pipeline_summary(self) -> None:
        self.pane.update_pipeline_summary()

    def set_diagnostics_visible(self, visible: bool) -> None:
        self.pane.set_diagnostics_visible(visible)

    def teardown(self) -> None:
        self.pane.teardown()


MainWindow = ModuleWindow


__all__ = [
    "BrightnessContrastDialog",
    "CropDialog",
    "GammaDialog",
    "MainWindow",
    "ModuleWindow",
    "NoiseReductionDialog",
    "NormalizeDialog",
    "PreprocessingPane",
    "SelectChannelDialog",
    "SharpenDialog",
]
