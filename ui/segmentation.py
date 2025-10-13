"""Qt widgets for the segmentation application."""
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
from PyQt5 import QtCore, QtGui, QtWidgets

from core.app_core import AppCore
from core.io_manager import IOManager
from core.segmentation import Config, Loader, Preprocessor, parse_bool
from processing.segmentation_pipeline import (
    PipelineStep,
    ProcessingPipeline,
    build_segmentation_pipeline,
    build_segmentation_pipeline_from_dict,
    get_settings_snapshot,
)


def _build_segmentation_pipeline_metadata(settings: Mapping[str, Any]) -> Dict[str, Any]:
    order_value = settings.get("segmentation/order", "")
    order_list = [entry for entry in order_value.split(",") if entry]
    enabled_steps = [
        key[len("segmentation/") : -len("/enabled")]
        for key, value in settings.items()
        if key.startswith("segmentation/")
        and key.endswith("/enabled")
        and parse_bool(value)
    ]
    return {
        "stage": "segmentation",
        "order": order_list,
        "enabled": enabled_steps,
    }


#####################################
# 4. PIPELINE ORDER MANAGER
#####################################

class PipelineOrderManager:
    def __init__(self, settings: QtCore.QSettings):
        self.settings = settings
    def get_order(self) -> List[str]:
        order_str = self.settings.value("segmentation/order", "")
        return order_str.split(",") if order_str else []
    def set_order(self, order: List[str]):
        self.settings.setValue("segmentation/order", ",".join(order))
    def append_function(self, func_name: str):
        order = self.get_order()
        order.append(func_name)
        self.set_order(order)

#####################################
# 5. ERROR DIALOG & EXCEPTIONS
#####################################

def show_error_dialog(message: str):
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Error")
    layout = QtWidgets.QVBoxLayout(dlg)
    text_edit = QtWidgets.QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setPlainText(message)
    layout.addWidget(text_edit)
    btn_layout = QtWidgets.QHBoxLayout()
    copy_btn = QtWidgets.QPushButton("Copy")
    close_btn = QtWidgets.QPushButton("Close")
    btn_layout.addWidget(copy_btn)
    btn_layout.addWidget(close_btn)
    layout.addLayout(btn_layout)
    copy_btn.clicked.connect(lambda: QtGui.QGuiApplication.clipboard().setText(message))
    close_btn.clicked.connect(dlg.accept)
    dlg.exec_()

def excepthook(exc_type, exc_value, exc_traceback):
    err = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    show_error_dialog(err)

sys.excepthook = excepthook

#####################################
# 6. IMAGE DISPLAY WIDGET
#####################################

class ImageDisplayWidget(QtWidgets.QLabel):
    def __init__(self, use_rgb_format: bool = False):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self.use_rgb_format = use_rgb_format
        self.setMinimumSize(1,1)
    def set_image(self, image: np.ndarray):
        if len(image.shape)==2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.use_rgb_format:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
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
# 7. SEGMENTATION DIALOGS
#####################################
# (Dialogs for each segmentation method follow, similar to those in preprocessing, adapted for segmentation.)
# For brevity, all dialogs are fully implemented below.

class GlobalThresholdDialog(QtWidgets.QDialog):
    def __init__(self, threshold: int = 127, preview_callback: Optional[Callable[[int], None]] = None):
        super().__init__()
        self.setWindowTitle("Global Threshold Parameters")
        self.preview_callback = preview_callback
        self.initial_threshold = threshold
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(0,255)
        self.threshold_spin.setValue(threshold)
        form_layout.addRow("Threshold:", self.threshold_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.threshold_spin.valueChanged.connect(lambda: self.preview_callback(self.threshold_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> int:
        return self.threshold_spin.value()
    def reset_to_initial(self):
        self.threshold_spin.setValue(self.initial_threshold)

class OtsuThresholdDialog(QtWidgets.QDialog):
    def __init__(self, preview_callback: Optional[Callable[[], None]] = None):
        super().__init__()
        self.setWindowTitle("Otsu Thresholding")
        self.preview_callback = preview_callback
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Otsu thresholding will be applied automatically.")
        layout.addWidget(label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class AdaptiveThresholdDialog(QtWidgets.QDialog):
    def __init__(self, block_size: int = 11, C: int = 2, preview_callback: Optional[Callable[[int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Adaptive Threshold Parameters")
        self.preview_callback = preview_callback
        self.initial_block_size = block_size
        self.initial_C = C
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.block_size_spin = QtWidgets.QSpinBox()
        self.block_size_spin.setRange(3,101)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(block_size)
        self.C_spin = QtWidgets.QSpinBox()
        self.C_spin.setRange(-10,10)
        self.C_spin.setValue(C)
        form_layout.addRow("Block Size:", self.block_size_spin)
        form_layout.addRow("C:", self.C_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.block_size_spin.valueChanged.connect(lambda: self.preview_callback(self.block_size_spin.value(), self.C_spin.value()) if self.preview_callback else None)
        self.C_spin.valueChanged.connect(lambda: self.preview_callback(self.block_size_spin.value(), self.C_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int]:
        return self.block_size_spin.value(), self.C_spin.value()
    def reset_to_initial(self):
        self.block_size_spin.setValue(self.initial_block_size)
        self.C_spin.setValue(self.initial_C)

class EdgeBasedSegmentationDialog(QtWidgets.QDialog):
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150, aperture_size: int = 3, preview_callback: Optional[Callable[[int,int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Edge-based Segmentation Parameters")
        self.preview_callback = preview_callback
        self.initial_low = low_threshold
        self.initial_high = high_threshold
        self.initial_aperture = aperture_size
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.low_spin = QtWidgets.QSpinBox()
        self.low_spin.setRange(0,255)
        self.low_spin.setValue(low_threshold)
        self.high_spin = QtWidgets.QSpinBox()
        self.high_spin.setRange(0,255)
        self.high_spin.setValue(high_threshold)
        self.aperture_spin = QtWidgets.QSpinBox()
        self.aperture_spin.setRange(3,7)
        self.aperture_spin.setSingleStep(2)
        self.aperture_spin.setValue(aperture_size)
        form_layout.addRow("Low Threshold:", self.low_spin)
        form_layout.addRow("High Threshold:", self.high_spin)
        form_layout.addRow("Aperture Size:", self.aperture_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.low_spin.valueChanged.connect(lambda: self.preview_callback(self.low_spin.value(), self.high_spin.value(), self.aperture_spin.value()) if self.preview_callback else None)
        self.high_spin.valueChanged.connect(lambda: self.preview_callback(self.low_spin.value(), self.high_spin.value(), self.aperture_spin.value()) if self.preview_callback else None)
        self.aperture_spin.valueChanged.connect(lambda: self.preview_callback(self.low_spin.value(), self.high_spin.value(), self.aperture_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int,int]:
        return self.low_spin.value(), self.high_spin.value(), self.aperture_spin.value()
    def reset_to_initial(self):
        self.low_spin.setValue(self.initial_low)
        self.high_spin.setValue(self.initial_high)
        self.aperture_spin.setValue(self.initial_aperture)

class WatershedDialog(QtWidgets.QDialog):
    def __init__(self, kernel_size: int = 3, opening_iterations: int = 2, dilation_iterations: int = 3, distance_threshold_factor: float = 0.7, preview_callback: Optional[Callable[[int,int,int,float], None]] = None):
        super().__init__()
        self.setWindowTitle("Watershed Segmentation Parameters")
        self.preview_callback = preview_callback
        self.initial_kernel = kernel_size
        self.initial_opening = opening_iterations
        self.initial_dilation = dilation_iterations
        self.initial_factor = distance_threshold_factor
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.kernel_spin = QtWidgets.QSpinBox()
        self.kernel_spin.setRange(1,15)
        self.kernel_spin.setValue(kernel_size)
        self.opening_spin = QtWidgets.QSpinBox()
        self.opening_spin.setRange(1,10)
        self.opening_spin.setValue(opening_iterations)
        self.dilation_spin = QtWidgets.QSpinBox()
        self.dilation_spin.setRange(1,10)
        self.dilation_spin.setValue(dilation_iterations)
        self.factor_spin = QtWidgets.QDoubleSpinBox()
        self.factor_spin.setRange(0.1,1.0)
        self.factor_spin.setSingleStep(0.1)
        self.factor_spin.setValue(distance_threshold_factor)
        form_layout.addRow("Kernel Size:", self.kernel_spin)
        form_layout.addRow("Opening Iterations:", self.opening_spin)
        form_layout.addRow("Dilation Iterations:", self.dilation_spin)
        form_layout.addRow("Distance Threshold Factor:", self.factor_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.kernel_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_spin.value(), self.opening_spin.value(), self.dilation_spin.value(), self.factor_spin.value()) if self.preview_callback else None)
        self.opening_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_spin.value(), self.opening_spin.value(), self.dilation_spin.value(), self.factor_spin.value()) if self.preview_callback else None)
        self.dilation_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_spin.value(), self.opening_spin.value(), self.dilation_spin.value(), self.factor_spin.value()) if self.preview_callback else None)
        self.factor_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_spin.value(), self.opening_spin.value(), self.dilation_spin.value(), self.factor_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int,int,float]:
        return (self.kernel_spin.value(), self.opening_spin.value(), self.dilation_spin.value(), self.factor_spin.value())
    def reset_to_initial(self):
        self.kernel_spin.setValue(self.initial_kernel)
        self.opening_spin.setValue(self.initial_opening)
        self.dilation_spin.setValue(self.initial_dilation)
        self.factor_spin.setValue(self.initial_factor)

class SobelDialog(QtWidgets.QDialog):
    def __init__(self, ksize: int = 3, preview_callback: Optional[Callable[[int], None]] = None):
        super().__init__()
        self.setWindowTitle("Sobel Operator Parameters")
        self.preview_callback = preview_callback
        self.initial_ksize = ksize if ksize % 2 == 1 else ksize+1
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.ksize_spin = QtWidgets.QSpinBox()
        self.ksize_spin.setRange(1,31)
        self.ksize_spin.setSingleStep(2)
        self.ksize_spin.setValue(self.initial_ksize)
        form_layout.addRow("Kernel Size (odd):", self.ksize_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.ksize_spin.valueChanged.connect(lambda: self.preview_callback(self.ksize_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> int:
        return self.ksize_spin.value()
    def reset_to_initial(self):
        self.ksize_spin.setValue(self.initial_ksize)

class PrewittDialog(QtWidgets.QDialog):
    def __init__(self, preview_callback: Optional[Callable[[], None]] = None):
        super().__init__()
        self.setWindowTitle("Prewitt Operator")
        self.preview_callback = preview_callback
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Prewitt operator uses fixed kernels.")
        layout.addWidget(label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class LaplacianDialog(QtWidgets.QDialog):
    def __init__(self, ksize: int = 3, preview_callback: Optional[Callable[[int], None]] = None):
        super().__init__()
        self.setWindowTitle("Laplacian Operator Parameters")
        self.preview_callback = preview_callback
        self.initial_ksize = ksize if ksize % 2 == 1 else ksize+1
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.ksize_spin = QtWidgets.QSpinBox()
        self.ksize_spin.setRange(1,31)
        self.ksize_spin.setSingleStep(2)
        self.ksize_spin.setValue(self.initial_ksize)
        form_layout.addRow("Kernel Size (odd):", self.ksize_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.ksize_spin.valueChanged.connect(lambda: self.preview_callback(self.ksize_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> int:
        return self.ksize_spin.value()
    def reset_to_initial(self):
        self.ksize_spin.setValue(self.initial_ksize)

class RegionGrowingDialog(QtWidgets.QDialog):
    def __init__(self, seed_x: int = 50, seed_y: int = 50, tolerance: int = 10, preview_callback: Optional[Callable[[int,int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Region Growing Parameters")
        self.preview_callback = preview_callback
        self.initial_seed_x = seed_x
        self.initial_seed_y = seed_y
        self.initial_tolerance = tolerance
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.seed_x_spin = QtWidgets.QSpinBox()
        self.seed_x_spin.setRange(0,1000)
        self.seed_x_spin.setValue(seed_x)
        self.seed_y_spin = QtWidgets.QSpinBox()
        self.seed_y_spin.setRange(0,1000)
        self.seed_y_spin.setValue(seed_y)
        self.tolerance_spin = QtWidgets.QSpinBox()
        self.tolerance_spin.setRange(0,100)
        self.tolerance_spin.setValue(tolerance)
        form_layout.addRow("Seed X:", self.seed_x_spin)
        form_layout.addRow("Seed Y:", self.seed_y_spin)
        form_layout.addRow("Tolerance:", self.tolerance_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.seed_x_spin.valueChanged.connect(lambda: self.preview_callback(self.seed_x_spin.value(), self.seed_y_spin.value(), self.tolerance_spin.value()) if self.preview_callback else None)
        self.seed_y_spin.valueChanged.connect(lambda: self.preview_callback(self.seed_x_spin.value(), self.seed_y_spin.value(), self.tolerance_spin.value()) if self.preview_callback else None)
        self.tolerance_spin.valueChanged.connect(lambda: self.preview_callback(self.seed_x_spin.value(), self.seed_y_spin.value(), self.tolerance_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int,int]:
        return self.seed_x_spin.value(), self.seed_y_spin.value(), self.tolerance_spin.value()
    def reset_to_initial(self):
        self.seed_x_spin.setValue(self.initial_seed_x)
        self.seed_y_spin.setValue(self.initial_seed_y)
        self.tolerance_spin.setValue(self.initial_tolerance)

class RegionSplittingMergingDialog(QtWidgets.QDialog):
    def __init__(self, min_size: int = 16, std_thresh: float = 10, preview_callback: Optional[Callable[[int], None]] = None):
        super().__init__()
        self.setWindowTitle("Region Splitting/Merging Parameters")
        self.preview_callback = preview_callback
        self.initial_min_size = min_size
        self.initial_std_thresh = std_thresh
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.min_size_spin = QtWidgets.QSpinBox()
        self.min_size_spin.setRange(1,1000)
        self.min_size_spin.setValue(min_size)
        self.std_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.std_thresh_spin.setRange(0.1,100)
        self.std_thresh_spin.setSingleStep(0.5)
        self.std_thresh_spin.setValue(std_thresh)
        form_layout.addRow("Minimum Region Size:", self.min_size_spin)
        form_layout.addRow("Std. Deviation Threshold:", self.std_thresh_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.min_size_spin.valueChanged.connect(lambda: self.preview_callback(self.min_size_spin.value()) if self.preview_callback else None)
        self.std_thresh_spin.valueChanged.connect(lambda: self.preview_callback(self.min_size_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> int:
        return self.min_size_spin.value()
    def reset_to_initial(self):
        self.min_size_spin.setValue(self.initial_min_size)
        self.std_thresh_spin.setValue(self.initial_std_thresh)

class KMeansSegmentationDialog(QtWidgets.QDialog):
    def __init__(self, K: int = 2, seed: int = 42, preview_callback: Optional[Callable[[int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("K-Means Segmentation Parameters")
        self.preview_callback = preview_callback
        self.initial_K = K
        self.initial_seed = seed
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.K_spin = QtWidgets.QSpinBox()
        self.K_spin.setRange(2,10)
        self.K_spin.setValue(K)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0,10000)
        self.seed_spin.setValue(seed)
        form_layout.addRow("Number of Clusters (K):", self.K_spin)
        form_layout.addRow("Seed:", self.seed_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.K_spin.valueChanged.connect(lambda: self.preview_callback(self.K_spin.value(), self.seed_spin.value()) if self.preview_callback else None)
        self.seed_spin.valueChanged.connect(lambda: self.preview_callback(self.K_spin.value(), self.seed_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int]:
        return self.K_spin.value(), self.seed_spin.value()
    def reset_to_initial(self):
        self.K_spin.setValue(self.initial_K)
        self.seed_spin.setValue(self.initial_seed)

class FuzzyCMeansDialog(QtWidgets.QDialog):
    def __init__(self, K: int = 2, seed: int = 42, preview_callback: Optional[Callable[[int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Fuzzy C-Means Segmentation Parameters")
        self.preview_callback = preview_callback
        self.initial_K = K
        self.initial_seed = seed
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.K_spin = QtWidgets.QSpinBox()
        self.K_spin.setRange(2,10)
        self.K_spin.setValue(K)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0,10000)
        self.seed_spin.setValue(seed)
        form_layout.addRow("Number of Clusters (K):", self.K_spin)
        form_layout.addRow("Seed:", self.seed_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.K_spin.valueChanged.connect(lambda: self.preview_callback(self.K_spin.value(), self.seed_spin.value()) if self.preview_callback else None)
        self.seed_spin.valueChanged.connect(lambda: self.preview_callback(self.K_spin.value(), self.seed_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int]:
        return self.K_spin.value(), self.seed_spin.value()
    def reset_to_initial(self):
        self.K_spin.setValue(self.initial_K)
        self.seed_spin.setValue(self.initial_seed)

class MeanShiftDialog(QtWidgets.QDialog):
    def __init__(self, spatial_radius: int = 20, color_radius: int = 30, preview_callback: Optional[Callable[[int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Mean Shift Segmentation Parameters")
        self.preview_callback = preview_callback
        self.initial_spatial = spatial_radius
        self.initial_color = color_radius
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.spatial_spin = QtWidgets.QSpinBox()
        self.spatial_spin.setRange(1,100)
        self.spatial_spin.setValue(spatial_radius)
        self.color_spin = QtWidgets.QSpinBox()
        self.color_spin.setRange(1,100)
        self.color_spin.setValue(color_radius)
        form_layout.addRow("Spatial Radius:", self.spatial_spin)
        form_layout.addRow("Color Radius:", self.color_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.spatial_spin.valueChanged.connect(lambda: self.preview_callback(self.spatial_spin.value(), self.color_spin.value()) if self.preview_callback else None)
        self.color_spin.valueChanged.connect(lambda: self.preview_callback(self.spatial_spin.value(), self.color_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int]:
        return self.spatial_spin.value(), self.color_spin.value()
    def reset_to_initial(self):
        self.spatial_spin.setValue(self.initial_spatial)
        self.color_spin.setValue(self.initial_color)

class GMMDialog(QtWidgets.QDialog):
    def __init__(self, components: int = 2, seed: int = 42, preview_callback: Optional[Callable[[int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("GMM Segmentation Parameters")
        self.preview_callback = preview_callback
        self.initial_components = components
        self.initial_seed = seed
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.comp_spin = QtWidgets.QSpinBox()
        self.comp_spin.setRange(2,10)
        self.comp_spin.setValue(components)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0,10000)
        self.seed_spin.setValue(seed)
        form_layout.addRow("Components:", self.comp_spin)
        form_layout.addRow("Seed:", self.seed_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.comp_spin.valueChanged.connect(lambda: self.preview_callback(self.comp_spin.value(), self.seed_spin.value()) if self.preview_callback else None)
        self.seed_spin.valueChanged.connect(lambda: self.preview_callback(self.comp_spin.value(), self.seed_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,int]:
        return self.comp_spin.value(), self.seed_spin.value()
    def reset_to_initial(self):
        self.comp_spin.setValue(self.initial_components)
        self.seed_spin.setValue(self.initial_seed)

class GraphCutsDialog(QtWidgets.QDialog):
    def __init__(self, preview_callback: Optional[Callable[[], None]] = None):
        super().__init__()
        self.setWindowTitle("Graph Cuts Parameters")
        self.preview_callback = preview_callback
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Graph Cuts will use a rectangle covering the image.")
        layout.addWidget(label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class ActiveContourDialog(QtWidgets.QDialog):
    def __init__(self, iterations: int = 250, alpha: float = 0.015, beta: float = 10, gamma: float = 0.001, preview_callback: Optional[Callable[[int,float,float,float], None]] = None):
        super().__init__()
        self.setWindowTitle("Active Contour Parameters")
        self.preview_callback = preview_callback
        self.initial_iterations = iterations
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.initial_gamma = gamma
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.iter_spin = QtWidgets.QSpinBox()
        self.iter_spin.setRange(50,1000)
        self.iter_spin.setValue(iterations)
        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.alpha_spin.setRange(0.001,0.1)
        self.alpha_spin.setSingleStep(0.001)
        self.alpha_spin.setValue(alpha)
        self.beta_spin = QtWidgets.QDoubleSpinBox()
        self.beta_spin.setRange(1,20)
        self.beta_spin.setValue(beta)
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.0001,0.01)
        self.gamma_spin.setSingleStep(0.0001)
        self.gamma_spin.setValue(gamma)
        form_layout.addRow("Iterations:", self.iter_spin)
        form_layout.addRow("Alpha:", self.alpha_spin)
        form_layout.addRow("Beta:", self.beta_spin)
        form_layout.addRow("Gamma:", self.gamma_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.iter_spin.valueChanged.connect(lambda: self.preview_callback(self.iter_spin.value(), self.alpha_spin.value(), self.beta_spin.value(), self.gamma_spin.value()) if self.preview_callback else None)
        self.alpha_spin.valueChanged.connect(lambda: self.preview_callback(self.iter_spin.value(), self.alpha_spin.value(), self.beta_spin.value(), self.gamma_spin.value()) if self.preview_callback else None)
        self.beta_spin.valueChanged.connect(lambda: self.preview_callback(self.iter_spin.value(), self.alpha_spin.value(), self.beta_spin.value(), self.gamma_spin.value()) if self.preview_callback else None)
        self.gamma_spin.valueChanged.connect(lambda: self.preview_callback(self.iter_spin.value(), self.alpha_spin.value(), self.beta_spin.value(), self.gamma_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[int,float,float,float]:
        return self.iter_spin.value(), self.alpha_spin.value(), self.beta_spin.value(), self.gamma_spin.value()
    def reset_to_initial(self):
        self.iter_spin.setValue(self.initial_iterations)
        self.alpha_spin.setValue(self.initial_alpha)
        self.beta_spin.setValue(self.initial_beta)
        self.gamma_spin.setValue(self.initial_gamma)

class OpeningDialog(QtWidgets.QDialog):
    def __init__(self, kernel_shape: str = "Rectangular", kernel_size: int = 3, iterations: int = 1, preview_callback: Optional[Callable[[str,int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Opening Parameters")
        self.preview_callback = preview_callback
        self.initial_kernel_shape = kernel_shape
        self.initial_kernel_size = kernel_size
        self.initial_iterations = iterations
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.kernel_shape_combo = QtWidgets.QComboBox()
        self.kernel_shape_combo.addItems(["Rectangular", "Elliptical", "Cross"])
        self.kernel_shape_combo.setCurrentText(kernel_shape)
        self.kernel_size_spin = QtWidgets.QSpinBox()
        self.kernel_size_spin.setRange(1,31)
        self.kernel_size_spin.setValue(kernel_size)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1,10)
        self.iterations_spin.setValue(iterations)
        form_layout.addRow("Kernel Shape:", self.kernel_shape_combo)
        form_layout.addRow("Kernel Size:", self.kernel_size_spin)
        form_layout.addRow("Iterations:", self.iterations_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.kernel_shape_combo.currentTextChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.kernel_size_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.iterations_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[str,int,int]:
        return self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()
    def reset_to_initial(self):
        self.kernel_shape_combo.setCurrentText(self.initial_kernel_shape)
        self.kernel_size_spin.setValue(self.initial_kernel_size)
        self.iterations_spin.setValue(self.initial_iterations)

class ClosingDialog(QtWidgets.QDialog):
    def __init__(self, kernel_shape: str = "Rectangular", kernel_size: int = 3, iterations: int = 1, preview_callback: Optional[Callable[[str,int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Closing Parameters")
        self.preview_callback = preview_callback
        self.initial_kernel_shape = kernel_shape
        self.initial_kernel_size = kernel_size
        self.initial_iterations = iterations
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.kernel_shape_combo = QtWidgets.QComboBox()
        self.kernel_shape_combo.addItems(["Rectangular", "Elliptical", "Cross"])
        self.kernel_shape_combo.setCurrentText(kernel_shape)
        self.kernel_size_spin = QtWidgets.QSpinBox()
        self.kernel_size_spin.setRange(1,31)
        self.kernel_size_spin.setValue(kernel_size)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1,10)
        self.iterations_spin.setValue(iterations)
        form_layout.addRow("Kernel Shape:", self.kernel_shape_combo)
        form_layout.addRow("Kernel Size:", self.kernel_size_spin)
        form_layout.addRow("Iterations:", self.iterations_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.kernel_shape_combo.currentTextChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.kernel_size_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.iterations_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[str,int,int]:
        return self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()
    def reset_to_initial(self):
        self.kernel_shape_combo.setCurrentText(self.initial_kernel_shape)
        self.kernel_size_spin.setValue(self.initial_kernel_size)
        self.iterations_spin.setValue(self.initial_iterations)

class DilationDialog(QtWidgets.QDialog):
    def __init__(self, kernel_shape: str = "Rectangular", kernel_size: int = 3, iterations: int = 1, preview_callback: Optional[Callable[[str,int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Dilation Parameters")
        self.preview_callback = preview_callback
        self.initial_kernel_shape = kernel_shape
        self.initial_kernel_size = kernel_size
        self.initial_iterations = iterations
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.kernel_shape_combo = QtWidgets.QComboBox()
        self.kernel_shape_combo.addItems(["Rectangular", "Elliptical", "Cross"])
        self.kernel_shape_combo.setCurrentText(kernel_shape)
        self.kernel_size_spin = QtWidgets.QSpinBox()
        self.kernel_size_spin.setRange(1,31)
        self.kernel_size_spin.setValue(kernel_size)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1,10)
        self.iterations_spin.setValue(iterations)
        form_layout.addRow("Kernel Shape:", self.kernel_shape_combo)
        form_layout.addRow("Kernel Size:", self.kernel_size_spin)
        form_layout.addRow("Iterations:", self.iterations_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.kernel_shape_combo.currentTextChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.kernel_size_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.iterations_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[str,int,int]:
        return self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()
    def reset_to_initial(self):
        self.kernel_shape_combo.setCurrentText(self.initial_kernel_shape)
        self.kernel_size_spin.setValue(self.initial_kernel_size)
        self.iterations_spin.setValue(self.initial_iterations)

class ErosionDialog(QtWidgets.QDialog):
    def __init__(self, kernel_shape: str = "Rectangular", kernel_size: int = 3, iterations: int = 1, preview_callback: Optional[Callable[[str,int,int], None]] = None):
        super().__init__()
        self.setWindowTitle("Erosion Parameters")
        self.preview_callback = preview_callback
        self.initial_kernel_shape = kernel_shape
        self.initial_kernel_size = kernel_size
        self.initial_iterations = iterations
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.kernel_shape_combo = QtWidgets.QComboBox()
        self.kernel_shape_combo.addItems(["Rectangular", "Elliptical", "Cross"])
        self.kernel_shape_combo.setCurrentText(kernel_shape)
        self.kernel_size_spin = QtWidgets.QSpinBox()
        self.kernel_size_spin.setRange(1,31)
        self.kernel_size_spin.setValue(kernel_size)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1,10)
        self.iterations_spin.setValue(iterations)
        form_layout.addRow("Kernel Shape:", self.kernel_shape_combo)
        form_layout.addRow("Kernel Size:", self.kernel_size_spin)
        form_layout.addRow("Iterations:", self.iterations_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.kernel_shape_combo.currentTextChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.kernel_size_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.iterations_spin.valueChanged.connect(lambda: self.preview_callback(self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> Tuple[str,int,int]:
        return self.kernel_shape_combo.currentText(), self.kernel_size_spin.value(), self.iterations_spin.value()
    def reset_to_initial(self):
        self.kernel_shape_combo.setCurrentText(self.initial_kernel_shape)
        self.kernel_size_spin.setValue(self.initial_kernel_size)
        self.iterations_spin.setValue(self.initial_iterations)

class BorderRemovalDialog(QtWidgets.QDialog):
    def __init__(self, border_distance: int = 100, preview_callback: Optional[Callable[[int], None]] = None):
        super().__init__()
        self.setWindowTitle("Border Removal Parameters")
        self.preview_callback = preview_callback
        self.initial_border_distance = border_distance
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.border_spin = QtWidgets.QSpinBox()
        self.border_spin.setRange(0,999)
        self.border_spin.setValue(border_distance)
        form_layout.addRow("Border Distance (pixels):", self.border_spin)
        layout.addLayout(form_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.border_spin.valueChanged.connect(lambda: self.preview_callback(self.border_spin.value()) if self.preview_callback else None)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    def get_values(self) -> int:
        return self.border_spin.value()
    def reset_to_initial(self):
        self.border_spin.setValue(self.initial_border_distance)

#####################################
# 8. MASS PROCESS FUNCTION (TOP-LEVEL)
#####################################
# This function is defined at module level for pickling.
def process_segmentation_file(
    file: str,
    folder: str,
    pipeline_settings: dict,
    pipeline_metadata: Mapping[str, Any],
    io_preferences: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    fullpath = os.path.join(folder, file)
    try:
        image = Loader.load_image(fullpath)
        pipeline = build_segmentation_pipeline_from_dict(pipeline_settings)
        processed = pipeline.apply(image)
        output_folder = os.path.join(folder, "segmented_output")
        os.makedirs(output_folder, exist_ok=True)
        name, ext = os.path.splitext(file)
        new_filename = name + "_seg" + ext
        outpath = os.path.join(output_folder, new_filename)
        io_manager = IOManager(io_preferences)
        io_manager.save_image(
            outpath,
            processed,
            metadata={
                "stage": "segmentation",
                "mode": "batch",
                "source": {"input": fullpath},
            },
            pipeline=pipeline_metadata,
            settings_snapshot=pipeline_settings,
        )
        return (file, True, "")
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return (file, False, str(e))

#####################################
# 9. MAIN WINDOW
#####################################

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_core: AppCore):
        super().__init__()
        self.app_core = app_core
        self.setWindowTitle("Image Segmentation Module")
        self.resize(1200,700)
        self.original_image: Optional[np.ndarray] = None
        self.segmentation_image: Optional[np.ndarray] = None
        self.base_image: Optional[np.ndarray] = None
        self.committed_image: Optional[np.ndarray] = None
        self.undo_stack: List[Tuple[np.ndarray, List[str]]] = []
        self.redo_stack: List[Tuple[np.ndarray, List[str]]] = []
        self.current_preview: Optional[np.ndarray] = None
        self.current_image_path: Optional[str] = None
        self.settings_manager = self.app_core.settings
        self.settings = self.settings_manager.backend
        # Set default segmentation parameters if missing.
        if not self.settings.contains("segmentation/Global/threshold"):
            self.settings.setValue("segmentation/Global/threshold", 127)
        # Disable all segmentation functions by default.
        for m in ["Global","Otsu","Adaptive","Edge","Watershed","Sobel","Prewitt",
                  "Laplacian","Region Growing","Region Splitting/Merging","K-Means",
                  "Fuzzy C-Means","Mean Shift","GMM","Graph Cuts","Active Contour",
                  "Opening","Closing","Dilation","Erosion","Border Removal"]:
            self.settings.setValue(f"segmentation/{m}/enabled", False)
        self.settings.setValue("segmentation/order", "")
        self.order_manager = PipelineOrderManager(self.settings)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        images_layout = QtWidgets.QHBoxLayout()
        # Original Image Display
        original_group = QtWidgets.QGroupBox("Original Image")
        orig_layout = QtWidgets.QVBoxLayout()
        self.original_display = ImageDisplayWidget(use_rgb_format=True)
        orig_scroll = QtWidgets.QScrollArea()
        orig_scroll.setWidgetResizable(True)
        orig_scroll.setWidget(self.original_display)
        orig_layout.addWidget(orig_scroll)
        original_group.setLayout(orig_layout)
        images_layout.addWidget(original_group)
        # Segmentation Preview Display
        preview_group = QtWidgets.QGroupBox("Segmentation Preview")
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
        self.pipeline = build_segmentation_pipeline(self.app_core)
        self.update_pipeline_label()
        if self.base_image is not None:
            self.update_preview()

    def update_pipeline_label(self):
        order = self.order_manager.get_order()
        self.pipeline_label.setText("Current Pipeline: " + " -> ".join(order) if order else "Current Pipeline: (none)")

    def rebuild_pipeline(self):
        self.pipeline = build_segmentation_pipeline(self.app_core)
        logging.debug("Pipeline rebuilt: " + ", ".join([step.name for step in self.pipeline.steps]))
        self.update_pipeline_label()

    def get_segmentation_order(self) -> List[str]:
        order_str = self.settings.value("segmentation/order", "")
        return order_str.split(",") if order_str else []

    def set_segmentation_order(self, order: List[str]):
        self.settings.setValue("segmentation/order", ",".join(order))
        self.update_pipeline_label()

    def commit_segmentation(self, func_name: str):
        self.order_manager.append_function(func_name)
        self.update_pipeline_label()

    def push_undo_state(self, backup: np.ndarray):
        self.undo_stack.append((backup.copy(), self.get_segmentation_order()))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            current_state = (self.committed_image.copy(), self.get_segmentation_order())
            self.redo_stack.append(current_state)
            prev_image, prev_order = self.undo_stack.pop()
            self.committed_image = prev_image.copy()
            self.set_segmentation_order(prev_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def redo(self):
        if self.redo_stack:
            current_state = (self.committed_image.copy(), self.get_segmentation_order())
            self.undo_stack.append(current_state)
            next_image, next_order = self.redo_stack.pop()
            self.committed_image = next_image.copy()
            self.set_segmentation_order(next_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.redo_stack) > 0)

    def build_menu(self):
        menubar = self.menuBar()
        # File Menu
        file_menu = menubar.addMenu("File")
        load_action = QtWidgets.QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        save_action = QtWidgets.QAction("Save Segmented Image", self)
        save_action.triggered.connect(self.save_segmented_image)
        file_menu.addAction(save_action)
        mass_action = QtWidgets.QAction("Mass Process Folder", self)
        mass_action.triggered.connect(self.mass_process)
        file_menu.addAction(mass_action)
        imp_action = QtWidgets.QAction("Import Segmentation Settings", self)
        imp_action.triggered.connect(self.import_pipeline)
        file_menu.addAction(imp_action)
        exp_action = QtWidgets.QAction("Export Segmentation Settings", self)
        exp_action.triggered.connect(self.export_pipeline)
        file_menu.addAction(exp_action)
        # Edit Menu
        edit_menu = menubar.addMenu("Edit")
        self.undo_action = QtWidgets.QAction("Undo", self)
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QtWidgets.QAction("Redo", self)
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)
        reset_action = QtWidgets.QAction("Reset All", self)
        reset_action.triggered.connect(self.reset_all)
        edit_menu.addAction(reset_action)
        # Segmentation Menu (Tree Structure)
        seg_menu = menubar.addMenu("Segmentation")
        # Thresholding Submenu
        thresh_menu = seg_menu.addMenu("Thresholding")
        global_action = QtWidgets.QAction("Global Threshold", self)
        global_action.triggered.connect(self.show_global_threshold_dialog)
        thresh_menu.addAction(global_action)
        otsu_action = QtWidgets.QAction("Otsu Threshold", self)
        otsu_action.triggered.connect(self.show_otsu_threshold_dialog)
        thresh_menu.addAction(otsu_action)
        adaptive_action = QtWidgets.QAction("Adaptive Threshold", self)
        adaptive_action.triggered.connect(self.show_adaptive_threshold_dialog)
        thresh_menu.addAction(adaptive_action)
        # Edge Detection Submenu
        edge_menu = seg_menu.addMenu("Edge Detection")
        edge_action = QtWidgets.QAction("Edge-based Segmentation", self)
        edge_action.triggered.connect(self.show_edge_based_dialog)
        edge_menu.addAction(edge_action)
        sobel_action = QtWidgets.QAction("Sobel Operator", self)
        sobel_action.triggered.connect(self.show_sobel_dialog)
        edge_menu.addAction(sobel_action)
        prewitt_action = QtWidgets.QAction("Prewitt Operator", self)
        prewitt_action.triggered.connect(self.show_prewitt_dialog)
        edge_menu.addAction(prewitt_action)
        laplacian_action = QtWidgets.QAction("Laplacian Operator", self)
        laplacian_action.triggered.connect(self.show_laplacian_dialog)
        edge_menu.addAction(laplacian_action)
        # Region-based Submenu
        region_menu = seg_menu.addMenu("Region-based")
        region_grow_action = QtWidgets.QAction("Region Growing", self)
        region_grow_action.triggered.connect(self.show_region_growing_dialog)
        region_menu.addAction(region_grow_action)
        region_split_action = QtWidgets.QAction("Region Splitting/Merging", self)
        region_split_action.triggered.connect(self.show_region_splitting_merging_dialog)
        region_menu.addAction(region_split_action)
        # Clustering Submenu
        clustering_menu = seg_menu.addMenu("Clustering")
        kmeans_action = QtWidgets.QAction("K-Means Segmentation", self)
        kmeans_action.triggered.connect(self.show_kmeans_dialog)
        clustering_menu.addAction(kmeans_action)
        fuzzy_action = QtWidgets.QAction("Fuzzy C-Means Segmentation", self)
        fuzzy_action.triggered.connect(self.show_fuzzy_dialog)
        clustering_menu.addAction(fuzzy_action)
        meanshift_action = QtWidgets.QAction("Mean Shift Segmentation", self)
        meanshift_action.triggered.connect(self.show_meanshift_dialog)
        clustering_menu.addAction(meanshift_action)
        gmm_action = QtWidgets.QAction("GMM Segmentation", self)
        gmm_action.triggered.connect(self.show_gmm_dialog)
        clustering_menu.addAction(gmm_action)
        # Other Submenu
        other_menu = seg_menu.addMenu("Other")
        graphcuts_action = QtWidgets.QAction("Graph Cuts", self)
        graphcuts_action.triggered.connect(self.show_graphcuts_dialog)
        other_menu.addAction(graphcuts_action)
        active_contour_action = QtWidgets.QAction("Active Contour", self)
        active_contour_action.triggered.connect(self.show_active_contour_dialog)
        other_menu.addAction(active_contour_action)
        # Morphological Operations Submenu
        morph_menu = seg_menu.addMenu("Morphological Operations")
        opening_action = QtWidgets.QAction("Opening", self)
        opening_action.triggered.connect(self.show_opening_dialog)
        morph_menu.addAction(opening_action)
        closing_action = QtWidgets.QAction("Closing", self)
        closing_action.triggered.connect(self.show_closing_dialog)
        morph_menu.addAction(closing_action)
        dilation_action = QtWidgets.QAction("Dilation", self)
        dilation_action.triggered.connect(self.show_dilation_dialog)
        morph_menu.addAction(dilation_action)
        erosion_action = QtWidgets.QAction("Erosion", self)
        erosion_action.triggered.connect(self.show_erosion_dialog)
        morph_menu.addAction(erosion_action)
        border_action = QtWidgets.QAction("Border Removal", self)
        border_action.triggered.connect(self.show_border_removal_dialog)
        morph_menu.addAction(border_action)

    def reset_all(self):
        default_methods = ["Global","Otsu","Adaptive","Edge","Watershed","Sobel","Prewitt",
                           "Laplacian","Region Growing","Region Splitting/Merging","K-Means",
                           "Fuzzy C-Means","Mean Shift","GMM","Graph Cuts","Active Contour",
                           "Opening","Closing","Dilation","Erosion","Border Removal"]
        for m in default_methods:
            self.settings.setValue(f"segmentation/{m}/enabled", False)
        self.settings.setValue("segmentation/order", "")
        self.order_manager.set_order([])
        self.rebuild_pipeline()
        if self.base_image is not None:
            self.committed_image = self.base_image.copy()
            self.current_preview = self.base_image.copy()
            self.preview_display.set_image(self.current_preview)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_actions()
        self.statusBar().showMessage("Reset to original settings.")

    def preview_update(self, func_name: str, new_params: Dict[str, Any]):
        temp_dict = get_settings_snapshot(self.settings_manager)
        mapping = {
            "Global": "Global",
            "Otsu": "Otsu",
            "Adaptive": "Adaptive",
            "Edge": "Edge",
            "Watershed": "Watershed",
            "Sobel": "Sobel",
            "Prewitt": "Prewitt",
            "Laplacian": "Laplacian",
            "Region Growing": "Region Growing",
            "Region Splitting/Merging": "Region Splitting/Merging",
            "K-Means": "K-Means",
            "Fuzzy C-Means": "Fuzzy C-Means",
            "Mean Shift": "Mean Shift",
            "GMM": "GMM",
            "Graph Cuts": "Graph Cuts",
            "Active Contour": "Active Contour",
            "Opening": "Opening",
            "Closing": "Closing",
            "Dilation": "Dilation",
            "Erosion": "Erosion",
            "Border Removal": "Border Removal"
        }
        ns = mapping.get(func_name, func_name)
        temp_dict[f"segmentation/{ns}/enabled"] = True
        order = temp_dict.get("segmentation/order", "")
        order_list = order.split(",") if order else []
        temp_order = order_list + [func_name]
        temp_dict["segmentation/order"] = ",".join(temp_order)
        for key, value in new_params.items():
            temp_dict[f"segmentation/{ns}/{key}"] = value
        temp_pipeline = build_segmentation_pipeline_from_dict(temp_dict, self.app_core)
        if self.base_image is not None:
            new_preview = temp_pipeline.apply(self.base_image)
            self.current_preview = new_preview.copy()
            self.preview_display.set_image(new_preview)

    def show_global_threshold_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_thresh = int(self.settings.value("segmentation/Global/threshold", 127))
        dlg = GlobalThresholdDialog(threshold=current_thresh,
                                    preview_callback=lambda t: self.preview_update("Global", {"threshold": t}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_thresh = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Global")
            self.settings.setValue("segmentation/Global/enabled", True)
            self.settings.setValue("segmentation/Global/threshold", new_thresh)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_otsu_threshold_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        dlg = OtsuThresholdDialog(preview_callback=lambda: self.preview_update("Otsu", {}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.push_undo_state(backup)
            self.commit_segmentation("Otsu")
            self.settings.setValue("segmentation/Otsu/enabled", True)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_adaptive_threshold_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_block = int(self.settings.value("segmentation/Adaptive/block_size", 11))
        current_C = int(self.settings.value("segmentation/Adaptive/C", 2))
        dlg = AdaptiveThresholdDialog(block_size=current_block, C=current_C,
                                      preview_callback=lambda b, C: self.preview_update("Adaptive", {"block_size": b, "C": C}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_block, new_C = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Adaptive")
            self.settings.setValue("segmentation/Adaptive/enabled", True)
            self.settings.setValue("segmentation/Adaptive/block_size", new_block)
            self.settings.setValue("segmentation/Adaptive/C", new_C)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_edge_based_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_low = int(self.settings.value("segmentation/Edge/low_threshold", 50))
        current_high = int(self.settings.value("segmentation/Edge/high_threshold", 150))
        current_aperture = int(self.settings.value("segmentation/Edge/aperture_size", 3))
        dlg = EdgeBasedSegmentationDialog(low_threshold=current_low, high_threshold=current_high, aperture_size=current_aperture,
                                          preview_callback=lambda l, h, a: self.preview_update("Edge", {"low_threshold": l, "high_threshold": h, "aperture_size": a}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_low, new_high, new_aperture = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Edge")
            self.settings.setValue("segmentation/Edge/enabled", True)
            self.settings.setValue("segmentation/Edge/low_threshold", new_low)
            self.settings.setValue("segmentation/Edge/high_threshold", new_high)
            self.settings.setValue("segmentation/Edge/aperture_size", new_aperture)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_watershed_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_kernel = int(self.settings.value("segmentation/Watershed/kernel_size", 3))
        current_opening = int(self.settings.value("segmentation/Watershed/opening_iterations", 2))
        current_dilation = int(self.settings.value("segmentation/Watershed/dilation_iterations", 3))
        current_factor = float(self.settings.value("segmentation/Watershed/distance_threshold_factor", 0.7))
        dlg = WatershedDialog(kernel_size=current_kernel, opening_iterations=current_opening,
                              dilation_iterations=current_dilation, distance_threshold_factor=current_factor,
                              preview_callback=lambda k, o, d, f: self.preview_update("Watershed", {"kernel_size": k, "opening_iterations": o, "dilation_iterations": d, "distance_threshold_factor": f}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_k, new_o, new_d, new_f = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Watershed")
            self.settings.setValue("segmentation/Watershed/enabled", True)
            self.settings.setValue("segmentation/Watershed/kernel_size", new_k)
            self.settings.setValue("segmentation/Watershed/opening_iterations", new_o)
            self.settings.setValue("segmentation/Watershed/dilation_iterations", new_d)
            self.settings.setValue("segmentation/Watershed/distance_threshold_factor", new_f)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_sobel_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_ksize = int(self.settings.value("segmentation/Sobel/ksize", 3))
        dlg = SobelDialog(ksize=current_ksize,
                          preview_callback=lambda k: self.preview_update("Sobel", {"ksize": k}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_k = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Sobel")
            self.settings.setValue("segmentation/Sobel/enabled", True)
            self.settings.setValue("segmentation/Sobel/ksize", new_k)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_prewitt_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        dlg = PrewittDialog(preview_callback=lambda: self.preview_update("Prewitt", {}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.push_undo_state(backup)
            self.commit_segmentation("Prewitt")
            self.settings.setValue("segmentation/Prewitt/enabled", True)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_laplacian_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_ksize = int(self.settings.value("segmentation/Laplacian/ksize", 3))
        dlg = LaplacianDialog(ksize=current_ksize,
                              preview_callback=lambda k: self.preview_update("Laplacian", {"ksize": k}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_k = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Laplacian")
            self.settings.setValue("segmentation/Laplacian/enabled", True)
            self.settings.setValue("segmentation/Laplacian/ksize", new_k)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_region_growing_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_seed_x = int(self.settings.value("segmentation/Region Growing/seed_x", 50))
        current_seed_y = int(self.settings.value("segmentation/Region Growing/seed_y", 50))
        current_tol = int(self.settings.value("segmentation/Region Growing/tolerance", 10))
        dlg = RegionGrowingDialog(seed_x=current_seed_x, seed_y=current_seed_y, tolerance=current_tol,
                                  preview_callback=lambda sx, sy, t: self.preview_update("Region Growing", {"seed": (sx, sy), "tolerance": t}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_sx, new_sy, new_t = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Region Growing")
            self.settings.setValue("segmentation/Region Growing/enabled", True)
            self.settings.setValue("segmentation/Region Growing/seed_x", new_sx)
            self.settings.setValue("segmentation/Region Growing/seed_y", new_sy)
            self.settings.setValue("segmentation/Region Growing/tolerance", new_t)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_region_splitting_merging_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_min = int(self.settings.value("segmentation/Region Splitting/Merging/min_size", 16))
        current_std = float(self.settings.value("segmentation/Region Splitting/Merging/std_thresh", 10))
        dlg = RegionSplittingMergingDialog(min_size=current_min, std_thresh=current_std,
                                           preview_callback=lambda m: self.preview_update("Region Splitting/Merging", {"min_size": m}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_min = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Region Splitting/Merging")
            self.settings.setValue("segmentation/Region Splitting/Merging/enabled", True)
            self.settings.setValue("segmentation/Region Splitting/Merging/min_size", new_min)
            self.settings.setValue("segmentation/Region Splitting/Merging/std_thresh", current_std)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_kmeans_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_K = int(self.settings.value("segmentation/K-Means/K", 2))
        current_seed = int(self.settings.value("segmentation/K-Means/seed", 42))
        dlg = KMeansSegmentationDialog(K=current_K, seed=current_seed,
                                       preview_callback=lambda K, s: self.preview_update("K-Means", {"K": K, "seed": s}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_K, new_seed = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("K-Means")
            self.settings.setValue("segmentation/K-Means/enabled", True)
            self.settings.setValue("segmentation/K-Means/K", new_K)
            self.settings.setValue("segmentation/K-Means/seed", new_seed)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_fuzzy_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_K = int(self.settings.value("segmentation/Fuzzy C-Means/K", 2))
        current_seed = int(self.settings.value("segmentation/Fuzzy C-Means/seed", 42))
        dlg = FuzzyCMeansDialog(K=current_K, seed=current_seed,
                                preview_callback=lambda K, s: self.preview_update("Fuzzy C-Means", {"K": K, "seed": s}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_K, new_seed = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Fuzzy C-Means")
            self.settings.setValue("segmentation/Fuzzy C-Means/enabled", True)
            self.settings.setValue("segmentation/Fuzzy C-Means/K", new_K)
            self.settings.setValue("segmentation/Fuzzy C-Means/seed", new_seed)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_meanshift_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_spatial = int(self.settings.value("segmentation/Mean Shift/spatial_radius", 20))
        current_color = int(self.settings.value("segmentation/Mean Shift/color_radius", 30))
        dlg = MeanShiftDialog(spatial_radius=current_spatial, color_radius=current_color,
                              preview_callback=lambda s, c: self.preview_update("Mean Shift", {"spatial_radius": s, "color_radius": c}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_spatial, new_color = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Mean Shift")
            self.settings.setValue("segmentation/Mean Shift/enabled", True)
            self.settings.setValue("segmentation/Mean Shift/spatial_radius", new_spatial)
            self.settings.setValue("segmentation/Mean Shift/color_radius", new_color)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_gmm_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_components = int(self.settings.value("segmentation/GMM/components", 2))
        current_seed = int(self.settings.value("segmentation/GMM/seed", 42))
        dlg = GMMDialog(components=current_components, seed=current_seed,
                        preview_callback=lambda comp, s: self.preview_update("GMM", {"components": comp, "seed": s}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_comp, new_seed = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("GMM")
            self.settings.setValue("segmentation/GMM/enabled", True)
            self.settings.setValue("segmentation/GMM/components", new_comp)
            self.settings.setValue("segmentation/GMM/seed", new_seed)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_graphcuts_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        dlg = GraphCutsDialog(preview_callback=lambda: self.preview_update("Graph Cuts", {}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.push_undo_state(backup)
            self.commit_segmentation("Graph Cuts")
            self.settings.setValue("segmentation/Graph Cuts/enabled", True)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_active_contour_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_iter = int(self.settings.value("segmentation/Active Contour/iterations", 250))
        current_alpha = float(self.settings.value("segmentation/Active Contour/alpha", 0.015))
        current_beta = float(self.settings.value("segmentation/Active Contour/beta", 10))
        current_gamma = float(self.settings.value("segmentation/Active Contour/gamma", 0.001))
        dlg = ActiveContourDialog(iterations=current_iter, alpha=current_alpha, beta=current_beta, gamma=current_gamma,
                                  preview_callback=lambda it, a, b, g: self.preview_update("Active Contour", {"iterations": it, "alpha": a, "beta": b, "gamma": g}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_iter, new_alpha, new_beta, new_gamma = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Active Contour")
            self.settings.setValue("segmentation/Active Contour/enabled", True)
            self.settings.setValue("segmentation/Active Contour/iterations", new_iter)
            self.settings.setValue("segmentation/Active Contour/alpha", new_alpha)
            self.settings.setValue("segmentation/Active Contour/beta", new_beta)
            self.settings.setValue("segmentation/Active Contour/gamma", new_gamma)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_opening_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_shape = str(self.settings.value("segmentation/Opening/kernel_shape", "Rectangular"))
        current_kernel = int(self.settings.value("segmentation/Opening/kernel_size", 3))
        current_iter = int(self.settings.value("segmentation/Opening/iterations", 1))
        dlg = OpeningDialog(kernel_shape=current_shape, kernel_size=current_kernel, iterations=current_iter,
                            preview_callback=lambda s, k, it: self.preview_update("Opening", {"kernel_shape": s, "kernel_size": k, "iterations": it}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_shape, new_kernel, new_iter = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Opening")
            self.settings.setValue("segmentation/Opening/enabled", True)
            self.settings.setValue("segmentation/Opening/kernel_shape", new_shape)
            self.settings.setValue("segmentation/Opening/kernel_size", new_kernel)
            self.settings.setValue("segmentation/Opening/iterations", new_iter)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_closing_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_shape = str(self.settings.value("segmentation/Closing/kernel_shape", "Rectangular"))
        current_kernel = int(self.settings.value("segmentation/Closing/kernel_size", 3))
        current_iter = int(self.settings.value("segmentation/Closing/iterations", 1))
        dlg = ClosingDialog(kernel_shape=current_shape, kernel_size=current_kernel, iterations=current_iter,
                            preview_callback=lambda s, k, it: self.preview_update("Closing", {"kernel_shape": s, "kernel_size": k, "iterations": it}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_shape, new_kernel, new_iter = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Closing")
            self.settings.setValue("segmentation/Closing/enabled", True)
            self.settings.setValue("segmentation/Closing/kernel_shape", new_shape)
            self.settings.setValue("segmentation/Closing/kernel_size", new_kernel)
            self.settings.setValue("segmentation/Closing/iterations", new_iter)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_dilation_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_shape = str(self.settings.value("segmentation/Dilation/kernel_shape", "Rectangular"))
        current_kernel = int(self.settings.value("segmentation/Dilation/kernel_size", 3))
        current_iter = int(self.settings.value("segmentation/Dilation/iterations", 1))
        dlg = DilationDialog(kernel_shape=current_shape, kernel_size=current_kernel, iterations=current_iter,
                             preview_callback=lambda s, k, it: self.preview_update("Dilation", {"kernel_shape": s, "kernel_size": k, "iterations": it}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_shape, new_kernel, new_iter = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Dilation")
            self.settings.setValue("segmentation/Dilation/enabled", True)
            self.settings.setValue("segmentation/Dilation/kernel_shape", new_shape)
            self.settings.setValue("segmentation/Dilation/kernel_size", new_kernel)
            self.settings.setValue("segmentation/Dilation/iterations", new_iter)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_erosion_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_shape = str(self.settings.value("segmentation/Erosion/kernel_shape", "Rectangular"))
        current_kernel = int(self.settings.value("segmentation/Erosion/kernel_size", 3))
        current_iter = int(self.settings.value("segmentation/Erosion/iterations", 1))
        dlg = ErosionDialog(kernel_shape=current_shape, kernel_size=current_kernel, iterations=current_iter,
                            preview_callback=lambda s, k, it: self.preview_update("Erosion", {"kernel_shape": s, "kernel_size": k, "iterations": it}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_shape, new_kernel, new_iter = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Erosion")
            self.settings.setValue("segmentation/Erosion/enabled", True)
            self.settings.setValue("segmentation/Erosion/kernel_shape", new_shape)
            self.settings.setValue("segmentation/Erosion/kernel_size", new_kernel)
            self.settings.setValue("segmentation/Erosion/iterations", new_iter)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def show_border_removal_dialog(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current_border = int(self.settings.value("segmentation/Border Removal/border_distance", 100))
        dlg = BorderRemovalDialog(border_distance=current_border,
                                  preview_callback=lambda b: self.preview_update("Border Removal", {"border_distance": b}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_border = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_segmentation("Border Removal")
            self.settings.setValue("segmentation/Border Removal/enabled", True)
            self.settings.setValue("segmentation/Border Removal/border_distance", new_border)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            if backup is not None:
                self.preview_display.set_image(backup)

    def load_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.bmp *.tiff *.npy)")
        if filename:
            try:
                self.original_image = Loader.load_image(filename)
                self.base_image = self.original_image.copy()
                self.committed_image = build_segmentation_pipeline(self.app_core).apply(self.base_image)
                self.current_preview = self.committed_image.copy()
                self.current_image_path = filename
                self.original_display.set_image(self.original_image)
                self.preview_display.set_image(self.current_preview)
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.update_undo_redo_actions()
                self.statusBar().showMessage("Image loaded.")
                # Reset segmentation toggles.
                for m in ["Global","Otsu","Adaptive","Edge","Watershed","Sobel","Prewitt",
                          "Laplacian","Region Growing","Region Splitting/Merging","K-Means",
                          "Fuzzy C-Means","Mean Shift","GMM","Graph Cuts","Active Contour",
                          "Opening","Closing","Dilation","Erosion","Border Removal"]:
                    self.settings.setValue(f"segmentation/{m}/enabled", False)
                self.settings.setValue("segmentation/order", "")
                self.order_manager.set_order([])
                self.rebuild_pipeline()
            except Exception as e:
                logging.exception("Error loading image.")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def update_preview(self):
        if self.base_image is None:
            return
        new_preview = self.pipeline.apply(self.base_image)
        self.committed_image = new_preview.copy()
        self.current_preview = new_preview.copy()
        self.preview_display.set_image(new_preview)

    def save_segmented_image(self):
        if self.current_preview is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No segmented image to save.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Segmented Image", "", "Image Files (*.bmp *.png *.jpg *.tiff *.npy)")
        if filename:
            try:
                io_manager = self.app_core.io_manager
                pipeline_settings = get_settings_snapshot(self.settings_manager)
                pipeline_metadata = _build_segmentation_pipeline_metadata(pipeline_settings)
                metadata: Dict[str, Any] = {
                    "stage": "segmentation",
                    "mode": "single",
                }
                if self.current_image_path:
                    metadata["source"] = {"input": self.current_image_path}
                result = io_manager.save_image(
                    filename,
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
            except Exception as e:
                logging.exception("Error saving image.")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save image:\n{e}")

    def mass_process(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder for Mass Processing")
        if not folder:
            return
        output_folder = os.path.join(folder, "segmented_output")
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in Config.SUPPORTED_FORMATS]
        if not files:
            QtWidgets.QMessageBox.information(self, "Mass Processing", "No supported image files found in the selected folder.")
            return
        # Capture pipeline settings into a dictionary for processing.
        pipeline_settings = get_settings_snapshot(self.settings_manager)
        pipeline_metadata = _build_segmentation_pipeline_metadata(pipeline_settings)
        io_preferences = self.app_core.io_manager.export_preferences()
        processed_count = 0
        errors = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_segmentation_file,
                    f,
                    folder,
                    pipeline_settings,
                    pipeline_metadata,
                    io_preferences,
                ): f
                for f in files
            }
            for future in concurrent.futures.as_completed(futures):
                file, success, err = future.result()
                if success:
                    processed_count += 1
                else:
                    errors.append(f"{file}: {err}")
        msg = f"Processed {processed_count} images.\nOutput folder: {output_folder}"
        if errors:
            msg += "\nSome errors occurred:\n" + "\n".join(errors)
        QtWidgets.QMessageBox.information(self, "Mass Processing", msg)

    def export_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Pipeline Settings", "", "JSON Files (*.json)")
        if filename:
            try:
                payload = self.settings_manager.to_json(
                    prefix="segmentation/", strip_prefix=True
                )
                Path(filename).write_text(payload, encoding="utf-8")
                QtWidgets.QMessageBox.information(self, "Export Pipeline", "Pipeline settings exported successfully.")
            except Exception as e:
                logging.exception("Error exporting pipeline.")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export pipeline:\n{e}")

    def import_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Pipeline Settings", "", "JSON Files (*.json)")
        if filename:
            try:
                payload = Path(filename).read_text(encoding="utf-8")
                self.settings_manager.from_json(
                    payload, prefix="segmentation/", clear=True
                )
                self.rebuild_pipeline()
                if self.base_image is not None:
                    self.committed_image = self.pipeline.apply(self.base_image)
                    self.current_preview = self.committed_image.copy()
                    self.preview_display.set_image(self.current_preview)
                QtWidgets.QMessageBox.information(self, "Import Pipeline", "Pipeline settings imported successfully.")
            except Exception as e:
                logging.exception("Error importing pipeline.")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to import pipeline:\n{e}")

#####################################
# 10. MAIN ENTRY POINT
#####################################

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    try:
        sys.exit(app.exec_())
    except Exception:
        logging.exception("Application encountered an error.")

if __name__ == "__main__":
    main()


__all__ = ["MainWindow", "PipelineOrderManager", "ImageDisplayWidget"]
