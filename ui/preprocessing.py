"""Qt widgets for the preprocessing application."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore
from core.preprocessing import Config, Loader
from core.thread_controller import OperationCancelled, ThreadController
from processing.pipeline_manager import PipelineManager
from processing.preprocessing_pipeline import (
    PreprocessingPipeline,
    build_preprocessing_pipeline,
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
    def __init__(self, alpha: float = 1.0, beta: int = 0, preview_callback: Optional[Callable[[float, int], None]] = None):
        super().__init__()
        self.setWindowTitle("Brightness / Contrast")
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.alpha_spin.value(), self.beta_spin.value())

    def get_values(self) -> Tuple[float, int]:
        return self.alpha_spin.value(), self.beta_spin.value()

    def reset_to_initial(self):
        self.alpha_spin.setValue(self.initial_alpha)
        self.beta_spin.setValue(self.initial_beta)


class GammaDialog(QtWidgets.QDialog):
    def __init__(self, gamma: float = 1.0, preview_callback: Optional[Callable[[float], None]] = None):
        super().__init__()
        self.setWindowTitle("Gamma Correction")
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.gamma_spin.value())

    def get_value(self) -> float:
        return self.gamma_spin.value()

    def reset_to_initial(self):
        self.gamma_spin.setValue(self.initial_gamma)


class NormalizeDialog(QtWidgets.QDialog):
    def __init__(self, alpha: int = 0, beta: int = 255, preview_callback: Optional[Callable[[int, int], None]] = None):
        super().__init__()
        self.setWindowTitle("Intensity Normalization")
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.alpha_spin.value(), self.beta_spin.value())

    def get_values(self) -> Tuple[int, int]:
        return self.alpha_spin.value(), self.beta_spin.value()

    def reset_to_initial(self):
        self.alpha_spin.setValue(self.initial_alpha)
        self.beta_spin.setValue(self.initial_beta)


class NoiseReductionDialog(QtWidgets.QDialog):
    def __init__(self, method: str = "Gaussian", ksize: int = 5, preview_callback: Optional[Callable[[str, int], None]] = None):
        super().__init__()
        self.setWindowTitle("Noise Reduction")
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.method_combo.currentText(), self.ksize_spin.value())

    def get_values(self) -> Tuple[str, int]:
        return self.method_combo.currentText(), self.ksize_spin.value()

    def reset_to_initial(self):
        self.method_combo.setCurrentText(self.initial_method)
        self.ksize_spin.setValue(self.initial_ksize)


class SharpenDialog(QtWidgets.QDialog):
    def __init__(self, strength: float = 1.0, preview_callback: Optional[Callable[[float], None]] = None):
        super().__init__()
        self.setWindowTitle("Sharpen")
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.strength_spin.value())

    def get_value(self) -> float:
        return self.strength_spin.value()

    def reset_to_initial(self):
        self.strength_spin.setValue(self.initial_strength)


class SelectChannelDialog(QtWidgets.QDialog):
    def __init__(self, current_channel: str = "All", preview_callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.setWindowTitle("Select Color Channel")
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.channel_combo.currentText())

    def get_value(self) -> str:
        return self.channel_combo.currentText()

    def reset_to_initial(self):
        self.channel_combo.setCurrentText(self.initial_channel)


class CropDialog(QtWidgets.QDialog):
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
        self.preview_callback = preview_callback
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

    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(
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
    def __init__(self, app_core: AppCore):
        super().__init__()
        self.app_core = app_core
        self.setWindowTitle("Image Pre-Processing Module")
        self.resize(1200, 700)
        self.original_image: Optional[np.ndarray] = None
        self.base_image: Optional[np.ndarray] = None
        self.committed_image: Optional[np.ndarray] = None
        self.current_preview: Optional[np.ndarray] = None
        self.pipeline_manager: PipelineManager = (
            self.app_core.get_preprocessing_pipeline_manager()
        )
        self.pipeline_manager.reset()
        if self.app_core.thread_controller is None:
            self.app_core.thread_controller = ThreadController(parent=self)
        self.thread_controller: ThreadController = self.app_core.thread_controller
        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._register_thread_signals()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        images_layout = QtWidgets.QHBoxLayout()
        original_group = QtWidgets.QGroupBox("Original Image")
        orig_layout = QtWidgets.QVBoxLayout()
        self.original_display = ImageDisplayWidget(use_rgb_format=True)
        orig_scroll = QtWidgets.QScrollArea()
        orig_scroll.setWidgetResizable(True)
        orig_scroll.setWidget(self.original_display)
        orig_layout.addWidget(orig_scroll)
        original_group.setLayout(orig_layout)
        images_layout.addWidget(original_group)

        preview_group = QtWidgets.QGroupBox("Pre-Processing Preview")
        prev_layout = QtWidgets.QVBoxLayout()
        self.preview_display = ImageDisplayWidget(use_rgb_format=False)
        prev_scroll = QtWidgets.QScrollArea()
        prev_scroll.setWidgetResizable(True)
        prev_scroll.setWidget(self.preview_display)
        prev_layout.addWidget(prev_scroll)
        preview_group.setLayout(prev_layout)
        images_layout.addWidget(preview_group)
        main_layout.addLayout(images_layout)

        self.pipeline_label = QtWidgets.QLabel("Current Pipeline: (none)")
        main_layout.addWidget(self.pipeline_label)
        self.statusBar().showMessage("Ready")

        self.build_menu()

        self.undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.redo)

        self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
        self.update_pipeline_label()
        if self.base_image is not None:
            self.update_preview()

    def update_pipeline_label(self):
        order = [step.name for step in self.pipeline_manager.iter_enabled_steps()]
        text = "Current Pipeline: " + " -> ".join(order) if order else "Current Pipeline: (none)"
        self.pipeline_label.setText(text)

    def rebuild_pipeline(self):
        self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
        logging.debug("Pipeline rebuilt with steps: " + ", ".join(step.name for step in self.pipeline.steps))
        self.update_pipeline_label()

    def undo(self):
        snapshot = self.pipeline_manager.undo(current_image=self.committed_image)
        if snapshot is None:
            return
        if snapshot.image is not None:
            self.committed_image = snapshot.image.copy()
        self.rebuild_pipeline()
        if self.base_image is not None:
            self._apply_pipeline_async(
                description="Updating preview",
                on_finished=self._update_preview_from_result,
            )
        self.update_undo_redo_actions()

    def redo(self):
        snapshot = self.pipeline_manager.redo(current_image=self.committed_image)
        if snapshot is None:
            return
        if snapshot.image is not None:
            self.committed_image = snapshot.image.copy()
        self.rebuild_pipeline()
        if self.base_image is not None:
            self._apply_pipeline_async(
                description="Updating preview",
                on_finished=self._update_preview_from_result,
            )
        self.update_undo_redo_actions()

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(self.pipeline_manager.can_undo())
        self.redo_action.setEnabled(self.pipeline_manager.can_redo())

    def build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        load_action = QtWidgets.QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        save_action = QtWidgets.QAction("Save Pre-Processed Image", self)
        save_action.triggered.connect(self.save_processed_image)
        file_menu.addAction(save_action)

        mass_pp_action = QtWidgets.QAction("Mass Pre-Process Folder", self)
        mass_pp_action.triggered.connect(self.mass_preprocess)
        file_menu.addAction(mass_pp_action)

        imp_action = QtWidgets.QAction("Import Pipeline Settings", self)
        imp_action.triggered.connect(self.import_pipeline)
        file_menu.addAction(imp_action)

        exp_action = QtWidgets.QAction("Export Pipeline Settings", self)
        exp_action.triggered.connect(self.export_pipeline)
        file_menu.addAction(exp_action)

        edit_menu = menubar.addMenu("Edit")
        self.undo_action = QtWidgets.QAction("Undo", self)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QtWidgets.QAction("Redo", self)
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

    def _register_thread_signals(self) -> None:
        self._current_task_description: str = ""
        self.thread_controller.task_started.connect(self._on_task_started)
        self.thread_controller.task_progress.connect(self._on_task_progress)
        self.thread_controller.task_finished.connect(self._on_task_finished)
        self.thread_controller.task_canceled.connect(self._on_task_canceled)
        self.thread_controller.task_failed.connect(self._on_task_failed)

    @QtCore.pyqtSlot(str)
    def _on_task_started(self, description: str) -> None:
        self._current_task_description = description or "Processing"
        self.statusBar().showMessage(self._current_task_description)

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

    @QtCore.pyqtSlot()
    def _on_task_canceled(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        self.statusBar().showMessage("Operation canceled", 2000)

    @QtCore.pyqtSlot(Exception, str)
    def _on_task_failed(self, error: Exception, stack: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        logging.error("Pipeline execution failed: %s\n%s", error, stack)
        QtWidgets.QMessageBox.critical(
            self,
            "Processing Error",
            f"{self._current_task_description} failed:\n{error}",
        )
        self.statusBar().showMessage("Error during processing", 4000)

    def _apply_pipeline_async(
        self,
        *,
        pipeline: Optional[PipelineManager] = None,
        image: Optional[np.ndarray] = None,
        description: str = "Processing image",
        on_finished: Optional[Callable[[np.ndarray], None]] = None,
        on_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        if pipeline is None:
            pipeline = self.pipeline
        if image is None:
            if self.base_image is None:
                return
            image = self.base_image

        def _handle_finished(result: np.ndarray) -> None:
            if on_finished is not None:
                on_finished(result)

        def _handle_canceled() -> None:
            if on_canceled is not None:
                on_canceled()

        self.thread_controller.run_pipeline(
            pipeline,
            image,
            description=description,
            on_finished=_handle_finished,
            on_canceled=_handle_canceled,
        )

    def _update_preview_from_result(self, image: np.ndarray) -> None:
        self.current_preview = image.copy()
        self.preview_display.set_image(self.current_preview)

    def _update_committed_from_result(self, image: np.ndarray) -> None:
        self.committed_image = image.copy()
        self.current_preview = self.committed_image.copy()
        self.preview_display.set_image(self.current_preview)
        self.update_undo_redo_actions()

        reset_action = QtWidgets.QAction("Reset All", self)
        reset_action.triggered.connect(self.reset_all)
        edit_menu.addAction(reset_action)

        pre_menu = menubar.addMenu("Pre-Processing")

        toggle_gray = QtWidgets.QAction("Toggle Greyscale", self)
        toggle_gray.triggered.connect(self.toggle_grayscale)
        pre_menu.addAction(toggle_gray)

        select_channel = QtWidgets.QAction("Select Color Channel", self)
        select_channel.triggered.connect(self.show_select_channel_dialog)
        pre_menu.addAction(select_channel)

        bc_action = QtWidgets.QAction("Brightness / Contrast", self)
        bc_action.triggered.connect(self.show_brightness_contrast_dialog)
        pre_menu.addAction(bc_action)

        gamma_action = QtWidgets.QAction("Gamma Correction", self)
        gamma_action.triggered.connect(self.show_gamma_dialog)
        pre_menu.addAction(gamma_action)

        norm_action = QtWidgets.QAction("Intensity Normalization", self)
        norm_action.triggered.connect(self.show_normalize_dialog)
        pre_menu.addAction(norm_action)

        noise_action = QtWidgets.QAction("Noise Reduction", self)
        noise_action.triggered.connect(self.show_noise_reduction_dialog)
        pre_menu.addAction(noise_action)

        sharpen_action = QtWidgets.QAction("Sharpen", self)
        sharpen_action.triggered.connect(self.show_sharpen_dialog)
        pre_menu.addAction(sharpen_action)

        crop_action = QtWidgets.QAction("Crop", self)
        crop_action.triggered.connect(self.show_crop_dialog)
        pre_menu.addAction(crop_action)

    def reset_all(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        self.pipeline_manager.push_state(image=backup)
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

    def _on_reset_completed(self, image: np.ndarray) -> None:
        self._update_committed_from_result(image)
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
                    cv2.imwrite(outpath, result)
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
                    image=None if self.committed_image is None else self.committed_image.copy()
                )
                self.app_core.load_preprocessing_pipeline(data)
                self.rebuild_pipeline()
                if self.base_image is not None:
                    def _applied(image: np.ndarray) -> None:
                        self._update_committed_from_result(image)
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
            self, "Load Image", "", "Image Files (*.jpg *.png *.tiff *.bmp)"
        )
        if not filename:
            return
        try:
            image = Loader.load_image(filename)
        except Exception as exc:  # pragma: no cover - user feedback
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {exc}")
            return

        self.original_image = image.copy()
        self.base_image = image.copy()
        self.committed_image = image.copy()
        self.pipeline = build_preprocessing_pipeline(self.app_core, self.pipeline_manager)
        self.original_display.set_image(self.original_image)
        self.preview_display.set_image(self.base_image)
        self._apply_pipeline_async(
            image=self.base_image,
            description="Generating preview",
            on_finished=self._update_preview_from_result,
        )
        self.pipeline_manager.clear_history()
        self.update_undo_redo_actions()
        self.statusBar().showMessage(f"Loaded image: {filename}")

    def save_processed_image(self):
        if self.committed_image is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "No processed image to save.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Pre-Processed Image", "", "Image Files (*.jpg *.png *.tiff *.bmp)"
        )
        if filename:
            cv2.imwrite(filename, self.committed_image)
            self.statusBar().showMessage(f"Saved processed image to: {filename}")

    def update_preview(self):
        if self.base_image is not None:
            self._apply_pipeline_async(
                description="Updating preview",
                on_finished=self._update_preview_from_result,
            )

    def toggle_grayscale(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        self.pipeline_manager.push_state(image=backup)
        self.pipeline_manager.toggle_step("Grayscale")
        self.rebuild_pipeline()
        if self.base_image is not None:
            self._apply_pipeline_async(
                description="Applying grayscale step",
                on_finished=self._update_committed_from_result,
            )
        self.update_undo_redo_actions()

    def preview_update(self, func_name: str, temp_params: Dict[str, Any]):
        if self.base_image is None:
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
        temp_pipeline = build_preprocessing_pipeline(self.app_core, temp_manager)
        self._apply_pipeline_async(
            pipeline=temp_pipeline,
            description="Updating preview",
            on_finished=self._update_preview_from_result,
        )

    def show_brightness_contrast_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("BrightnessContrast")
        current_alpha = float(step.params.get("alpha", 1.0))
        current_beta = int(step.params.get("beta", 0))
        dlg = BrightnessContrastDialog(
            alpha=current_alpha,
            beta=current_beta,
            preview_callback=lambda a, b: self.preview_update(
                "BrightnessContrast", {"alpha": a, "beta": b}
            ),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_alpha, new_beta = dlg.get_values()
            self.pipeline_manager.push_state(image=backup)
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
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_gamma_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("Gamma")
        current_gamma = float(step.params.get("gamma", 1.0))
        dlg = GammaDialog(
            gamma=current_gamma,
            preview_callback=lambda g: self.preview_update("Gamma", {"gamma": g}),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_gamma = dlg.get_value()
            self.pipeline_manager.push_state(image=backup)
            self.pipeline_manager.set_step_enabled("Gamma", True)
            self.pipeline_manager.update_step_params("Gamma", {"gamma": new_gamma})
            self.rebuild_pipeline()
            if self.base_image is not None:
                self._apply_pipeline_async(
                    description="Applying gamma correction",
                    on_finished=self._update_committed_from_result,
                )
            self.update_undo_redo_actions()
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_normalize_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("IntensityNormalization")
        current_alpha = int(step.params.get("alpha", 0))
        current_beta = int(step.params.get("beta", 255))
        dlg = NormalizeDialog(
            alpha=current_alpha,
            beta=current_beta,
            preview_callback=lambda a, b: self.preview_update(
                "IntensityNormalization", {"alpha": a, "beta": b}
            ),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_alpha, new_beta = dlg.get_values()
            self.pipeline_manager.push_state(image=backup)
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
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_noise_reduction_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("NoiseReduction")
        current_method = step.params.get("method", "Gaussian")
        current_ksize = int(step.params.get("ksize", 5))
        dlg = NoiseReductionDialog(
            method=current_method,
            ksize=current_ksize,
            preview_callback=lambda m, k: self.preview_update(
                "NoiseReduction", {"method": m, "ksize": k}
            ),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_method, new_ksize = dlg.get_values()
            self.pipeline_manager.push_state(image=backup)
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
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_sharpen_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("Sharpen")
        current_strength = float(step.params.get("strength", 1.0))
        dlg = SharpenDialog(
            strength=current_strength,
            preview_callback=lambda s: self.preview_update("Sharpen", {"strength": s}),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_strength = dlg.get_value()
            self.pipeline_manager.push_state(image=backup)
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
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_select_channel_dialog(self):
        backup = None if self.committed_image is None else self.committed_image.copy()
        step = self.pipeline_manager.get_step("SelectChannel")
        current_channel = step.params.get("channel", "All")
        dlg = SelectChannelDialog(
            current_channel=current_channel,
            preview_callback=lambda c: self.preview_update("SelectChannel", {"channel": c}),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_channel = dlg.get_value()
            self.pipeline_manager.push_state(image=backup)
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
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

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
            preview_callback=lambda x, y, w, h: self.preview_update(
                "Crop", {"x_offset": x, "y_offset": y, "width": w, "height": h}
            ),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_x, new_y, new_width, new_height = dlg.get_values()
            self.pipeline_manager.push_state(image=backup)
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
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

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
