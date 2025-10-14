"""Qt widgets for the extraction application."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore
from core.extraction import Config, Loader, Preprocessor, parse_bool
from core.path_sanitizer import PathValidationError, sanitize_user_path
from processing.extraction_pipeline import (
    PipelineStep,
    ProcessingPipeline,
    build_extraction_pipeline,
    build_extraction_pipeline_from_dict,
    get_extraction_settings_snapshot,
)

from yam_processor.ui.error_reporter import ErrorResolution, present_error_report


def _build_extraction_pipeline_metadata(settings: Mapping[str, Any]) -> Dict[str, Any]:
    order_value = settings.get("extraction/order", "")
    order_list = [entry for entry in order_value.split(",") if entry]
    enabled_steps = [
        key[len("extraction/") : -len("/enabled")]
        for key, value in settings.items()
        if key.startswith("extraction/")
        and key.endswith("/enabled")
        and parse_bool(value)
    ]
    return {
        "stage": "extraction",
        "order": order_list,
        "enabled": enabled_steps,
    }


LOGGER = logging.getLogger(__name__)


try:
    import pyperclip
except ImportError:  # pragma: no cover - optional dependency
    pyperclip = None

# 6. EXTRACTION PIPELINE ORDER MANAGER (allows duplicates)
#####################################

class ExtractionPipelineOrderManager:
    def __init__(self, settings: QtCore.QSettings):
        self.settings = settings
    def get_order(self) -> List[str]:
        order_str = self.settings.value("extraction/order", "")
        return order_str.split(",") if order_str else []
    def set_order(self, order: List[str]):
        self.settings.setValue("extraction/order", ",".join(order))
    def append_function(self, func_name: str):
        order = self.get_order()
        order.append(func_name)
        self.set_order(order)

#####################################
# 7. IMAGE DISPLAY WIDGET
#####################################

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

#####################################
# 8. EXTRACTION DIALOGS
#####################################

class RegionPropertiesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Region Properties Extraction")
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Extract region properties from the image.")
        layout.addWidget(label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class HuMomentsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hu Moments Extraction")
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Compute Hu invariant moments from the image.")
        layout.addWidget(label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class LBPDialog(QtWidgets.QDialog):
    def __init__(self, P=8, R=1.0, preview_callback: Optional[Callable[[int, float], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LBP Extraction Parameters")
        self.preview_callback = preview_callback
        self.initial_P = P
        self.initial_R = R
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.P_spin = QtWidgets.QSpinBox()
        self.P_spin.setRange(1, 24)
        self.P_spin.setValue(P)
        self.R_spin = QtWidgets.QDoubleSpinBox()
        self.R_spin.setRange(0.1, 10.0)
        self.R_spin.setValue(R)
        self.R_spin.setSingleStep(0.1)
        form_layout.addRow("Number of Points (P):", self.P_spin)
        form_layout.addRow("Radius (R):", self.R_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.P_spin.valueChanged.connect(self.on_value_changed)
        self.R_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.P_spin.value(), self.R_spin.value())
    def get_values(self) -> Tuple[int, float]:
        return self.P_spin.value(), self.R_spin.value()

class HaralickDialog(QtWidgets.QDialog):
    def __init__(self, distance=1, angle=0.0, preview_callback: Optional[Callable[[int, float], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Haralick Extraction Parameters")
        self.preview_callback = preview_callback
        self.initial_distance = distance
        self.initial_angle = angle
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.distance_spin = QtWidgets.QSpinBox()
        self.distance_spin.setRange(1, 10)
        self.distance_spin.setValue(distance)
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 3.14)
        self.angle_spin.setSingleStep(0.1)
        self.angle_spin.setValue(angle)
        form_layout.addRow("Distance:", self.distance_spin)
        form_layout.addRow("Angle (radians):", self.angle_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.distance_spin.valueChanged.connect(self.on_value_changed)
        self.angle_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.distance_spin.value(), self.angle_spin.value())
    def get_values(self) -> Tuple[int, float]:
        return self.distance_spin.value(), self.angle_spin.value()

class GaborDialog(QtWidgets.QDialog):
    def __init__(self, ksize=21, sigma=5.0, theta=0.0, lambd=10.0, gamma=0.5, psi=0.0,
                 preview_callback: Optional[Callable[[int, float, float, float, float, float], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gabor Extraction Parameters")
        self.preview_callback = preview_callback
        self.initial_ksize = ksize if ksize % 2 == 1 else ksize+1
        self.initial_sigma = sigma
        self.initial_theta = theta
        self.initial_lambd = lambd
        self.initial_gamma = gamma
        self.initial_psi = psi
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.ksize_spin = QtWidgets.QSpinBox()
        self.ksize_spin.setRange(3, 51)
        self.ksize_spin.setSingleStep(2)
        self.ksize_spin.setValue(self.initial_ksize)
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 20.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setValue(sigma)
        self.theta_spin = QtWidgets.QDoubleSpinBox()
        self.theta_spin.setRange(0.0, 3.14)
        self.theta_spin.setSingleStep(0.1)
        self.theta_spin.setValue(theta)
        self.lambd_spin = QtWidgets.QDoubleSpinBox()
        self.lambd_spin.setRange(1.0, 50.0)
        self.lambd_spin.setSingleStep(0.5)
        self.lambd_spin.setValue(lambd)
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(gamma)
        self.psi_spin = QtWidgets.QDoubleSpinBox()
        self.psi_spin.setRange(0.0, 3.14)
        self.psi_spin.setSingleStep(0.1)
        self.psi_spin.setValue(psi)
        form_layout.addRow("Kernel Size (odd):", self.ksize_spin)
        form_layout.addRow("Sigma:", self.sigma_spin)
        form_layout.addRow("Theta (radians):", self.theta_spin)
        form_layout.addRow("Wavelength:", self.lambd_spin)
        form_layout.addRow("Gamma:", self.gamma_spin)
        form_layout.addRow("Psi:", self.psi_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.ksize_spin.valueChanged.connect(self.on_value_changed)
        self.sigma_spin.valueChanged.connect(self.on_value_changed)
        self.theta_spin.valueChanged.connect(self.on_value_changed)
        self.lambd_spin.valueChanged.connect(self.on_value_changed)
        self.gamma_spin.valueChanged.connect(self.on_value_changed)
        self.psi_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.ksize_spin.value(), self.sigma_spin.value(),
                                  self.theta_spin.value(), self.lambd_spin.value(),
                                  self.gamma_spin.value(), self.psi_spin.value())
    def get_values(self) -> Tuple[int, float, float, float, float, float]:
        return (self.ksize_spin.value(), self.sigma_spin.value(), self.theta_spin.value(),
                self.lambd_spin.value(), self.gamma_spin.value(), self.psi_spin.value())

class FourierDialog(QtWidgets.QDialog):
    def __init__(self, num_coeff=10, preview_callback: Optional[Callable[[int], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fourier Extraction Parameters")
        self.preview_callback = preview_callback
        self.initial_num_coeff = num_coeff
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.coeff_spin = QtWidgets.QSpinBox()
        self.coeff_spin.setRange(2, 50)
        self.coeff_spin.setValue(num_coeff)
        form_layout.addRow("Number of Coefficients:", self.coeff_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.coeff_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.coeff_spin.value())
    def get_values(self) -> int:
        return self.coeff_spin.value()

class HOGDialog(QtWidgets.QDialog):
    def __init__(self, orientations=9, pixels_per_cell=8, cells_per_block=3,
                 preview_callback: Optional[Callable[[int, int, int], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HOG Extraction Parameters")
        self.preview_callback = preview_callback
        self.initial_orientations = orientations
        self.initial_ppc = pixels_per_cell
        self.initial_cpb = cells_per_block
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.orient_spin = QtWidgets.QSpinBox()
        self.orient_spin.setRange(1, 20)
        self.orient_spin.setValue(orientations)
        self.ppc_spin = QtWidgets.QSpinBox()
        self.ppc_spin.setRange(4, 32)
        self.ppc_spin.setValue(pixels_per_cell)
        self.cpb_spin = QtWidgets.QSpinBox()
        self.cpb_spin.setRange(1, 10)
        self.cpb_spin.setValue(cells_per_block)
        form_layout.addRow("Orientations:", self.orient_spin)
        form_layout.addRow("Pixels per Cell:", self.ppc_spin)
        form_layout.addRow("Cells per Block:", self.cpb_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.orient_spin.valueChanged.connect(self.on_value_changed)
        self.ppc_spin.valueChanged.connect(self.on_value_changed)
        self.cpb_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.orient_spin.value(), self.ppc_spin.value(), self.cpb_spin.value())
    def get_values(self) -> Tuple[int, int, int]:
        return self.orient_spin.value(), self.ppc_spin.value(), self.cpb_spin.value()

class HistogramStatsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram Statistics Extraction")
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Compute histogram statistics.")
        layout.addWidget(label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class FractalDialog(QtWidgets.QDialog):
    def __init__(self, min_box_size=2, preview_callback: Optional[Callable[[int], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fractal Dimension Parameters")
        self.preview_callback = preview_callback
        self.initial_min_box = min_box_size
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.box_spin = QtWidgets.QSpinBox()
        self.box_spin.setRange(1, 20)
        self.box_spin.setValue(min_box_size)
        form_layout.addRow("Min Box Size:", self.box_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.box_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.box_spin.value())
    def get_values(self) -> int:
        return self.box_spin.value()

class ApproximateShapeDialog(QtWidgets.QDialog):
    def __init__(self, error_threshold=1.0, preview_callback: Optional[Callable[[float], None]]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Approximate Shape Extraction Parameters")
        self.preview_callback = preview_callback
        self.initial_error = error_threshold
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.error_spin = QtWidgets.QDoubleSpinBox()
        self.error_spin.setRange(0.0001, 10.0)
        self.error_spin.setSingleStep(0.0001)
        self.error_spin.setDecimals(4)
        self.error_spin.setValue(error_threshold)
        form_layout.addRow("Error Threshold (px):", self.error_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.error_spin.valueChanged.connect(self.on_value_changed)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def on_value_changed(self):
        if self.preview_callback:
            self.preview_callback(self.error_spin.value())
    def get_values(self) -> float:
        return self.error_spin.value()

#####################################
# 9. MAIN WINDOW
#####################################

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_core: AppCore):
        super().__init__()
        self.app_core = app_core
        self.setWindowTitle("Microscopic Feature Extraction")
        self.resize(1200, 700)
        self.original_image: Optional[np.ndarray] = None
        self.base_image: Optional[np.ndarray] = None
        self.committed_image: Optional[np.ndarray] = None
        self.current_preview: Optional[np.ndarray] = None
        self.undo_stack: List[Tuple[np.ndarray, List[str]]] = []
        self.redo_stack: List[Tuple[np.ndarray, List[str]]] = []
        self.current_image_path: Optional[str] = None
        self.settings_manager = self.app_core.settings
        self.settings = self.settings_manager.backend
        # Set default extraction settings if not present
        default_methods = ["Region Properties", "Hu Moments", "LBP", "Haralick", "Gabor",
                           "Fourier", "HOG", "Histogram", "Fractal", "Approximate Shape"]
        defaults = {
            "Region Properties": {},
            "Hu Moments": {},
            "LBP": {"P": 8, "R": 1.0},
            "Haralick": {"distance": 1, "angle": 0.0},
            "Gabor": {"ksize": 21, "sigma": 5.0, "theta": 0.0, "lambd": 10.0, "gamma": 0.5, "psi": 0.0},
            "Fourier": {"num_coeff": 10},
            "HOG": {"orientations": 9, "ppc": 8, "cpb": 3},
            "Histogram": {},
            "Fractal": {"min_box_size": 2},
            "Approximate Shape": {"error_threshold": 1.0}
        }
        for m in default_methods:
            self.settings.setValue(f"extraction/{m}/enabled", False)
            for key, value in defaults[m].items():
                self.settings.setValue(f"extraction/{m}/{key}", value)
        self.settings.setValue("extraction/order", "")
        self.order_manager = ExtractionPipelineOrderManager(self.settings)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        images_layout = QtWidgets.QHBoxLayout()
        orig_group = QtWidgets.QGroupBox("Original Image")
        orig_layout = QtWidgets.QVBoxLayout()
        self.original_display = ImageDisplayWidget(use_rgb_format=True)
        orig_scroll = QtWidgets.QScrollArea()
        orig_scroll.setWidgetResizable(True)
        orig_scroll.setWidget(self.original_display)
        orig_layout.addWidget(orig_scroll)
        orig_group.setLayout(orig_layout)
        images_layout.addWidget(orig_group)
        feat_group = QtWidgets.QGroupBox("Feature Extraction Preview")
        feat_layout = QtWidgets.QVBoxLayout()
        self.preview_display = ImageDisplayWidget(use_rgb_format=False)
        feat_scroll = QtWidgets.QScrollArea()
        feat_scroll.setWidgetResizable(True)
        feat_scroll.setWidget(self.preview_display)
        feat_layout.addWidget(feat_scroll)
        feat_group.setLayout(feat_layout)
        images_layout.addWidget(feat_group)
        main_layout.addLayout(images_layout)
        self.pipeline_label = QtWidgets.QLabel("Current Pipeline: (none)")
        main_layout.addWidget(self.pipeline_label)
        self.statusBar().showMessage("Ready")
        self.build_menu()
        self.undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.redo)
        self.pipeline = build_extraction_pipeline(self.app_core)
        self.update_pipeline_label()
        if self.base_image is not None:
            self.update_preview()

    def update_pipeline_label(self):
        order = self.order_manager.get_order()
        self.pipeline_label.setText("Current Pipeline: " + " -> ".join(order) if order else "Current Pipeline: (none)")

    def rebuild_pipeline(self):
        self.pipeline = build_extraction_pipeline(self.app_core)
        logging.debug("Pipeline rebuilt with steps: " + ", ".join([step.name for step in self.pipeline.steps]))
        self.update_pipeline_label()

    def get_extraction_order(self) -> List[str]:
        order_str = self.settings.value("extraction/order", "")
        return order_str.split(",") if order_str else []

    def set_extraction_order(self, order: List[str]):
        self.settings.setValue("extraction/order", ",".join(order))
        self.update_pipeline_label()

    def commit_extraction(self, func_name: str):
        self.order_manager.append_function(func_name)
        self.update_pipeline_label()

    def push_undo_state(self, backup: np.ndarray):
        self.undo_stack.append((backup.copy(), self.get_extraction_order()))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            current_state = (self.committed_image.copy(), self.get_extraction_order())
            self.redo_stack.append(current_state)
            prev_image, prev_order = self.undo_stack.pop()
            self.committed_image = prev_image.copy()
            self.set_extraction_order(prev_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def redo(self):
        if self.redo_stack:
            current_state = (self.committed_image.copy(), self.get_extraction_order())
            self.undo_stack.append(current_state)
            next_image, next_order = self.redo_stack.pop()
            self.committed_image = next_image.copy()
            self.set_extraction_order(next_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.redo_stack) > 0)

    def reset_all(self):
        self.settings.setValue("extraction/Region Properties/enabled", False)
        self.settings.setValue("extraction/Hu Moments/enabled", False)
        self.settings.setValue("extraction/LBP/enabled", False)
        self.settings.setValue("extraction/Haralick/enabled", False)
        self.settings.setValue("extraction/Gabor/enabled", False)
        self.settings.setValue("extraction/Fourier/enabled", False)
        self.settings.setValue("extraction/HOG/enabled", False)
        self.settings.setValue("extraction/Histogram/enabled", False)
        self.settings.setValue("extraction/Fractal/enabled", False)
        self.settings.setValue("extraction/Approximate Shape/enabled", False)
        self.settings.setValue("extraction/LBP/P", 8)
        self.settings.setValue("extraction/LBP/R", 1.0)
        self.settings.setValue("extraction/Haralick/distance", 1)
        self.settings.setValue("extraction/Haralick/angle", 0.0)
        self.settings.setValue("extraction/Gabor/ksize", 21)
        self.settings.setValue("extraction/Gabor/sigma", 5.0)
        self.settings.setValue("extraction/Gabor/theta", 0.0)
        self.settings.setValue("extraction/Gabor/lambd", 10.0)
        self.settings.setValue("extraction/Gabor/gamma", 0.5)
        self.settings.setValue("extraction/Gabor/psi", 0.0)
        self.settings.setValue("extraction/Fourier/num_coeff", 10)
        self.settings.setValue("extraction/HOG/orientations", 9)
        self.settings.setValue("extraction/HOG/ppc", 8)
        self.settings.setValue("extraction/HOG/cpb", 3)
        self.settings.setValue("extraction/Fractal/min_box_size", 2)
        self.settings.setValue("extraction/Approximate Shape/error_threshold", 1.0)
        self.order_manager.set_order([])
        self.rebuild_pipeline()
        if self.base_image is not None:
            self.committed_image = self.base_image.copy()
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
            self.statusBar().showMessage("Reset all extraction settings to defaults.")

    def preview_update(self, func_name: str, new_params: Dict[str, Any]):
        temp_dict = get_extraction_settings_snapshot(self.settings_manager)
        # For simplicity, use the same method name for keys
        temp_dict[f"extraction/{func_name}/enabled"] = True
        order = temp_dict.get("extraction/order", "")
        order_list = order.split(",") if order else []
        temp_order = order_list + [func_name]
        temp_dict["extraction/order"] = ",".join(temp_order)
        for key, value in new_params.items():
            temp_dict[f"extraction/{func_name}/{key}"] = value
        temp_pipeline = build_extraction_pipeline_from_dict(temp_dict, self.app_core)
        if self.base_image is not None:
            new_preview = temp_pipeline.apply(self.base_image)
            self.current_preview = new_preview.copy()
            self.preview_display.set_image(new_preview)

    def build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_act = QtWidgets.QAction("Load Image", self)
        load_act.triggered.connect(self.load_image)
        file_menu.addAction(load_act)
        save_act = QtWidgets.QAction("Save Extracted Image", self)
        save_act.triggered.connect(self.save_processed_image)
        file_menu.addAction(save_act)
        mass_act = QtWidgets.QAction("Mass Extract Folder", self)
        mass_act.triggered.connect(self.mass_extract_folder)
        file_menu.addAction(mass_act)
        file_menu.addSeparator()
        export_regions_act = QtWidgets.QAction("Export Segmented Regions", self)
        export_regions_act.triggered.connect(self.export_regions)
        file_menu.addAction(export_regions_act)
        file_menu.addSeparator()
        export_data_act = QtWidgets.QAction("Export Extraction Data", self)
        export_data_act.triggered.connect(self.export_data)
        file_menu.addAction(export_data_act)
        mass_export_data_act = QtWidgets.QAction("Mass Export Extraction Data", self)
        mass_export_data_act.triggered.connect(self.mass_export_data)
        file_menu.addAction(mass_export_data_act)
        file_menu.addSeparator()
        imp_act = QtWidgets.QAction("Import Extraction Settings", self)
        imp_act.triggered.connect(self.import_settings)
        file_menu.addAction(imp_act)
        exp_act = QtWidgets.QAction("Export Extraction Settings", self)
        exp_act.triggered.connect(self.export_settings)
        file_menu.addAction(exp_act)
        edit_menu = menubar.addMenu("Edit")
        self.undo_action = QtWidgets.QAction("Undo", self)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QtWidgets.QAction("Redo", self)
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)
        reset_act = QtWidgets.QAction("Reset All", self)
        reset_act.triggered.connect(self.reset_all)
        edit_menu.addAction(reset_act)
        extract_menu = menubar.addMenu("Extraction")
        def label_with_order(method):
            order = self.get_extraction_order()
            if method in order:
                idx = order.index(method) + 1
                return f"{idx}. {method}"
            else:
                return method
        rp_act = QtWidgets.QAction(label_with_order("Region Properties"), self)
        rp_act.triggered.connect(self.extraction_region_properties)
        extract_menu.addAction(rp_act)
        hm_act = QtWidgets.QAction(label_with_order("Hu Moments"), self)
        hm_act.triggered.connect(self.extraction_hu_moments)
        extract_menu.addAction(hm_act)
        lbp_act = QtWidgets.QAction(label_with_order("LBP"), self)
        lbp_act.triggered.connect(self.extraction_lbp)
        extract_menu.addAction(lbp_act)
        har_act = QtWidgets.QAction(label_with_order("Haralick"), self)
        har_act.triggered.connect(self.extraction_haralick)
        extract_menu.addAction(har_act)
        gabor_act = QtWidgets.QAction(label_with_order("Gabor"), self)
        gabor_act.triggered.connect(self.extraction_gabor)
        extract_menu.addAction(gabor_act)
        fourier_act = QtWidgets.QAction(label_with_order("Fourier"), self)
        fourier_act.triggered.connect(self.extraction_fourier)
        extract_menu.addAction(fourier_act)
        hog_act = QtWidgets.QAction(label_with_order("HOG"), self)
        hog_act.triggered.connect(self.extraction_hog)
        extract_menu.addAction(hog_act)
        hist_act = QtWidgets.QAction(label_with_order("Histogram"), self)
        hist_act.triggered.connect(self.extraction_histogram)
        extract_menu.addAction(hist_act)
        fractal_act = QtWidgets.QAction(label_with_order("Fractal"), self)
        fractal_act.triggered.connect(self.extraction_fractal)
        extract_menu.addAction(fractal_act)
        approx_act = QtWidgets.QAction(label_with_order("Approximate Shape"), self)
        approx_act.triggered.connect(self.extraction_approximate_shape)
        extract_menu.addAction(approx_act)

    # Extraction Handlers
    def extraction_region_properties(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        dlg = RegionPropertiesDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.push_undo_state(backup)
            self.commit_extraction("Region Properties")
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_hu_moments(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        dlg = HuMomentsDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.push_undo_state(backup)
            self.commit_extraction("Hu Moments")
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_lbp(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_P = int(self.settings.value("extraction/LBP/P", 8))
        current_R = float(self.settings.value("extraction/LBP/R", 1.0))
        dlg = LBPDialog(P=current_P, R=current_R,
                        preview_callback=lambda p, r: self.preview_update("LBP", {"P": p, "R": r}),
                        parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            P, R = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("LBP")
            self.settings.setValue("extraction/LBP/P", P)
            self.settings.setValue("extraction/LBP/R", R)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_haralick(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_distance = int(self.settings.value("extraction/Haralick/distance", 1))
        current_angle = float(self.settings.value("extraction/Haralick/angle", 0.0))
        dlg = HaralickDialog(distance=current_distance, angle=current_angle,
                             preview_callback=lambda d, a: self.preview_update("Haralick", {"distance": d, "angle": a}),
                             parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            d, a = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("Haralick")
            self.settings.setValue("extraction/Haralick/distance", d)
            self.settings.setValue("extraction/Haralick/angle", a)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_gabor(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_ksize = int(self.settings.value("extraction/Gabor/ksize", 21))
        current_sigma = float(self.settings.value("extraction/Gabor/sigma", 5.0))
        current_theta = float(self.settings.value("extraction/Gabor/theta", 0.0))
        current_lambd = float(self.settings.value("extraction/Gabor/lambd", 10.0))
        current_gamma = float(self.settings.value("extraction/Gabor/gamma", 0.5))
        current_psi = float(self.settings.value("extraction/Gabor/psi", 0.0))
        dlg = GaborDialog(ksize=current_ksize, sigma=current_sigma, theta=current_theta,
                          lambd=current_lambd, gamma=current_gamma, psi=current_psi,
                          preview_callback=lambda ks, s, t, l, g, p: self.preview_update("Gabor", {"ksize": ks, "sigma": s, "theta": t, "lambd": l, "gamma": g, "psi": p}),
                          parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            ks, s, t, l, g, p = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("Gabor")
            self.settings.setValue("extraction/Gabor/ksize", ks)
            self.settings.setValue("extraction/Gabor/sigma", s)
            self.settings.setValue("extraction/Gabor/theta", t)
            self.settings.setValue("extraction/Gabor/lambd", l)
            self.settings.setValue("extraction/Gabor/gamma", g)
            self.settings.setValue("extraction/Gabor/psi", p)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_fourier(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_num = int(self.settings.value("extraction/Fourier/num_coeff", 10))
        dlg = FourierDialog(num_coeff=current_num,
                            preview_callback=lambda n: self.preview_update("Fourier", {"num_coeff": n}),
                            parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            num = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("Fourier")
            self.settings.setValue("extraction/Fourier/num_coeff", num)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_hog(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_orient = int(self.settings.value("extraction/HOG/orientations", 9))
        current_ppc = int(self.settings.value("extraction/HOG/ppc", 8))
        current_cpb = int(self.settings.value("extraction/HOG/cpb", 3))
        dlg = HOGDialog(orientations=current_orient, pixels_per_cell=current_ppc, cells_per_block=current_cpb,
                        preview_callback=lambda o, ppc, cpb: self.preview_update("HOG", {"orientations": o, "pixels_per_cell": (ppc, ppc), "cells_per_block": (cpb, cpb)}),
                        parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            o, ppc, cpb = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("HOG")
            self.settings.setValue("extraction/HOG/orientations", o)
            self.settings.setValue("extraction/HOG/ppc", ppc)
            self.settings.setValue("extraction/HOG/cpb", cpb)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_histogram(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        dlg = HistogramStatsDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.push_undo_state(backup)
            self.commit_extraction("Histogram")
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_fractal(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_box = int(self.settings.value("extraction/Fractal/min_box_size", 2))
        dlg = FractalDialog(min_box_size=current_box,
                            preview_callback=lambda b: self.preview_update("Fractal", {"min_box_size": b}),
                            parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            box = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("Fractal")
            self.settings.setValue("extraction/Fractal/min_box_size", box)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def extraction_approximate_shape(self):
        backup = self.current_preview.copy() if self.current_preview is not None else None
        current_error = float(self.settings.value("extraction/Approximate Shape/error_threshold", 1.0))
        dlg = ApproximateShapeDialog(error_threshold=current_error,
                                     preview_callback=lambda e: self.preview_update("Approximate Shape", {"error_threshold": e}),
                                     parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            error = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_extraction("Approximate Shape")
            self.settings.setValue("extraction/Approximate Shape/error_threshold", error)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

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
        metadata: Dict[str, object] = {"module": "extraction"}
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
                    extra={"component": "extraction", "error": str(exc)},
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
            except Exception as retry_error:  # pragma: no cover - user feedback
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

    def load_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.bmp *.tiff *.npy)"
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
            self.original_image = Loader.load_image(str(path))
            self.base_image = self.original_image.copy()
            self.committed_image = build_extraction_pipeline(self.app_core).apply(
                self.base_image
            )
            self.current_preview = self.committed_image.copy()
            self.current_image_path = str(path)
            self.original_display.set_image(self.original_image)
            self.preview_display.set_image(self.current_preview)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_undo_redo_actions()
            self.statusBar().showMessage("Image loaded.")
            self.settings.setValue("extraction/Region Properties/enabled", False)
            self.settings.setValue("extraction/Hu Moments/enabled", False)
            self.settings.setValue("extraction/LBP/enabled", False)
            self.settings.setValue("extraction/Haralick/enabled", False)
            self.settings.setValue("extraction/Gabor/enabled", False)
            self.settings.setValue("extraction/Fourier/enabled", False)
            self.settings.setValue("extraction/HOG/enabled", False)
            self.settings.setValue("extraction/Histogram/enabled", False)
            self.settings.setValue("extraction/Fractal/enabled", False)
            self.settings.setValue("extraction/Approximate Shape/enabled", False)
            self.order_manager.set_order([])
            self.rebuild_pipeline()

        try:
            _attempt_load()
        except Exception as error:
            self._report_error(
                self.tr("Failed to load image: {error}").format(error=error),
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

    def update_preview(self):
        if self.base_image is None:
            return
        new_preview = self.pipeline.apply(self.base_image)
        self.committed_image = new_preview.copy()
        self.current_preview = new_preview.copy()
        self.preview_display.set_image(new_preview)

    def save_processed_image(self):
        if self.current_preview is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No extracted image to save.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Extracted Image", "", "Image Files (*.bmp *.png *.jpg *.tiff *.npy)"
        )
        destination = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=False,
            operation="save_processed_image",
            window_title=self.tr("Save Image"),
            message_template=self.tr("The selected file path could not be used: {error}"),
        )
        if destination is None:
            return

        def _attempt_save() -> None:
            pipeline_settings = get_extraction_settings_snapshot(self.settings_manager)
            pipeline_metadata = _build_extraction_pipeline_metadata(pipeline_settings)
            metadata: Dict[str, Any] = {
                "stage": "extraction",
                "mode": "single",
            }
            if self.current_image_path:
                metadata["source"] = {"input": self.current_image_path}
            result = self.app_core.io_manager.save_image(
                destination,
                self.current_preview,
                metadata=metadata,
                pipeline=pipeline_metadata,
                settings_snapshot=pipeline_settings,
            )
            QtWidgets.QMessageBox.information(
                self,
                "Save Image",
                f"Image saved successfully to {result.image_path}",
            )

        try:
            _attempt_save()
        except Exception as error:
            self._report_error(
                self.tr("Failed to save image: {error}").format(error=error),
                context={
                    "operation": "save_processed_image",
                    "destination": str(destination),
                },
                window_title=self.tr("Save Image"),
                enable_retry=True,
                retry_label=self.tr("&Retry Save"),
                retry_callback=_attempt_save,
                fallback_traceback=traceback.format_exc(),
            )

    def mass_extract_folder(self):
        raw_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder for Mass Extraction"
        )
        folder_path = self._sanitize_dialog_path(
            raw_folder,
            allow_directory=True,
            allow_file=False,
            must_exist=True,
            operation="mass_extract_folder",
            window_title=self.tr("Mass Extraction"),
            message_template=self.tr("The selected folder could not be used: {error}"),
            extra_context={"dialog": "mass_extract"},
        )
        if folder_path is None:
            return
        output_folder = folder_path.parent / f"{folder_path.name}_feat"
        output_folder.mkdir(parents=True, exist_ok=True)
        pipeline_settings = get_extraction_settings_snapshot(self.settings_manager)
        pipeline_metadata = _build_extraction_pipeline_metadata(pipeline_settings)
        io_manager = self.app_core.io_manager
        count = 0
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix.lower() in Config.SUPPORTED_FORMATS:
                try:
                    image = Loader.load_image(str(path))
                    processed = self.pipeline.apply(image)
                    image_outpath = output_folder / f"{path.stem}_feat{path.suffix}"
                    io_manager.save_image(
                        image_outpath,
                        processed,
                        metadata={
                            "stage": "extraction",
                            "mode": "batch",
                            "source": {"input": str(path)},
                        },
                        pipeline=pipeline_metadata,
                        settings_snapshot=pipeline_settings,
                    )
                    self.export_all_extraction_data(processed, path.stem, output_folder)
                    count += 1
                except Exception as e:
                    logging.error("Failed to process %s: %s", path.name, e)
        QtWidgets.QMessageBox.information(
            self,
            "Mass Extraction",
            f"Processed {count} images.\nOutput folder: {output_folder}",
        )

    def export_regions(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        try:
            # Here we use a dummy path since the function only uses the basename
            count = export_segmented_regions(self.original_image, "dummy_path.png")
            QtWidgets.QMessageBox.information(self, "Export Segmented Regions", f"Exported {count} regions.")
        except Exception as error:
            self._report_error(
                self.tr("Failed to export regions: {error}").format(error=error),
                context={
                    "operation": "export_regions",
                    "source": self.current_image_path or "",
                },
                window_title=self.tr("Export Segmented Regions"),
                enable_retry=True,
                retry_label=self.tr("&Retry Export"),
                retry_callback=self.export_regions,
                fallback_traceback=traceback.format_exc(),
            )

    def export_data(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        base_name = "extracted_data"
        output_folder = Config.OUTPUT_DIR

        def _attempt_export() -> None:
            os.makedirs(output_folder, exist_ok=True)
            self.export_all_extraction_data(
                self.pipeline.apply(self.original_image), base_name, output_folder
            )
            QtWidgets.QMessageBox.information(
                self, "Export Extraction Data", f"Data exported to {output_folder}"
            )

        try:
            _attempt_export()
        except Exception as error:
            self._report_error(
                self.tr("Failed to export extraction data: {error}").format(error=error),
                context={
                    "operation": "export_extraction_data",
                    "destination": Path(output_folder),
                    "base_name": base_name,
                },
                window_title=self.tr("Export Extraction Data"),
                enable_retry=True,
                retry_label=self.tr("&Retry Export"),
                retry_callback=_attempt_export,
                fallback_traceback=traceback.format_exc(),
            )

    def mass_export_data(self):
        raw_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder for Mass Export Extraction Data"
        )
        folder_path = self._sanitize_dialog_path(
            raw_folder,
            allow_directory=True,
            allow_file=False,
            must_exist=True,
            operation="mass_export_data",
            window_title=self.tr("Mass Export Extraction Data"),
            message_template=self.tr("The selected folder could not be used: {error}"),
            extra_context={"dialog": "mass_export_data"},
        )
        if folder_path is None:
            return
        output_folder = folder_path.parent / f"{folder_path.name}_data"
        output_folder.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix.lower() in Config.SUPPORTED_FORMATS:
                try:
                    image = Loader.load_image(str(path))
                    base_name = path.stem
                    self.export_all_extraction_data(
                        self.pipeline.apply(image), base_name, output_folder
                    )
                    count += 1
                except Exception as e:
                    logging.error("Failed to export data for %s: %s", path.name, e)
        QtWidgets.QMessageBox.information(
            self,
            "Mass Export Extraction Data",
            f"Exported data for {count} images.\nOutput folder: {output_folder}",
        )

    def export_all_extraction_data(self, image: np.ndarray, base_filename: str, output_folder: str):
        order_str = self.settings.value("extraction/order", "")
        order = order_str.split(",") if order_str else []
        for method in order:
            if method == "Region Properties":
                df = region_properties_data(image)
            elif method == "Hu Moments":
                df = hu_moments_data(image)
            elif method == "LBP":
                P = int(self.settings.value("extraction/LBP/P", 8))
                R = float(self.settings.value("extraction/LBP/R", 1.0))
                df = lbp_data(image, P, R)
            elif method == "Haralick":
                d = int(self.settings.value("extraction/Haralick/distance", 1))
                a = float(self.settings.value("extraction/Haralick/angle", 0.0))
                df = haralick_data(image, d, a)
            elif method == "Gabor":
                ks = int(self.settings.value("extraction/Gabor/ksize", 21))
                sigma = float(self.settings.value("extraction/Gabor/sigma", 5.0))
                theta = float(self.settings.value("extraction/Gabor/theta", 0.0))
                lambd = float(self.settings.value("extraction/Gabor/lambd", 10.0))
                gamma_val = float(self.settings.value("extraction/Gabor/gamma", 0.5))
                psi = float(self.settings.value("extraction/Gabor/psi", 0.0))
                df = gabor_data(image, ks, sigma, theta, lambd, gamma_val, psi)
            elif method == "Fourier":
                num = int(self.settings.value("extraction/Fourier/num_coeff", 10))
                df = fourier_data(image, num)
            elif method == "HOG":
                orient = int(self.settings.value("extraction/HOG/orientations", 9))
                ppc = int(self.settings.value("extraction/HOG/ppc", 8))
                cpb = int(self.settings.value("extraction/HOG/cpb", 3))
                df = hog_data(image, orient, (ppc, ppc), (cpb, cpb))
            elif method == "Histogram":
                df = histogram_data(image)
            elif method == "Fractal":
                mbs = int(self.settings.value("extraction/Fractal/min_box_size", 2))
                df = fractal_data(image, mbs)
            elif method == "Approximate Shape":
                error = float(self.settings.value("extraction/Approximate Shape/error_threshold", 1.0))
                df = approximate_shape_data(image, error)
            else:
                continue
            filename = f"{base_filename}_{method}.csv"
            df.to_csv(os.path.join(output_folder, filename), index=False)

    def import_settings(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Extraction Settings", "", "JSON Files (*.json)"
        )
        source = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=True,
            operation="import_extraction_settings",
            window_title=self.tr("Settings Import"),
            message_template=self.tr("The selected file could not be used: {error}"),
        )
        if source is None:
            return

        def _attempt_import() -> None:
            payload = source.read_text(encoding="utf-8")
            self.settings_manager.from_json(
                payload, prefix="extraction/", clear=True
            )
            self.rebuild_pipeline()
            if self.base_image is not None:
                self.committed_image = self.pipeline.apply(self.base_image)
                self.current_preview = self.committed_image.copy()
                self.preview_display.set_image(self.current_preview)
            QtWidgets.QMessageBox.information(
                self, "Settings Import", "Extraction settings imported successfully."
            )

        try:
            _attempt_import()
        except Exception as error:
            self._report_error(
                self.tr("Failed to import extraction settings: {error}").format(error=error),
                context={
                    "operation": "import_extraction_settings",
                    "source": str(source),
                },
                window_title=self.tr("Settings Import"),
                enable_retry=True,
                retry_label=self.tr("&Retry Import"),
                retry_callback=_attempt_import,
                fallback_traceback=traceback.format_exc(),
            )

    def export_settings(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Extraction Settings", "", "JSON Files (*.json)"
        )
        destination = self._sanitize_dialog_path(
            filename,
            allow_directory=False,
            allow_file=True,
            must_exist=False,
            operation="export_extraction_settings",
            window_title=self.tr("Settings Export"),
            message_template=self.tr("The selected file path could not be used: {error}"),
        )
        if destination is None:
            return

        def _attempt_export() -> None:
            payload = self.settings_manager.to_json(
                prefix="extraction/", strip_prefix=True
            )
            destination.write_text(payload, encoding="utf-8")
            QtWidgets.QMessageBox.information(
                self, "Settings Export", "Extraction settings exported successfully."
            )

        try:
            _attempt_export()
        except Exception as error:
            self._report_error(
                self.tr("Failed to export extraction settings: {error}").format(error=error),
                context={
                    "operation": "export_extraction_settings",
                    "destination": str(destination),
                },
                window_title=self.tr("Settings Export"),
                enable_retry=True,
                retry_label=self.tr("&Retry Export"),
                retry_callback=_attempt_export,
                fallback_traceback=traceback.format_exc(),
            )

#####################################
# MAIN ENTRY POINT
#####################################

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


__all__ = ["MainWindow", "PipelineOrderManager", "ImageDisplayWidget"]
