#!/usr/bin/env python3
"""
Microscopic Image Processing App
=================================

This application processes microscopic images using a configurable pipeline.
It supports interactive live preview for adjustable functions and includes a
Pre‑Processing menu with the following items (in order):

  • Toggle Greyscale  
  • Brightness / Contrast  
  • Gamma Correction  
  • Intensity Normalization  
  • Noise Reduction  
  • Sharpen  
  • Select Color Channel  
  • Crop

Additionally, the File menu contains:
  • Load Image  
  • Save Pre‑Processed Image  
  • Mass Pre‑Process Folder  
  • Import Pipeline Settings  
  • Export Pipeline Settings  

In the Edit menu, a new “Reset All” function resets all function parameters to their
default state, clears the pipeline, and restores the preview to the original unaltered image.

When a function is selected, its dialog lets you adjust parameters. As you change values,
the dialog’s preview callback builds a temporary pipeline (with that function appended) and
applies it to the base (original) image so the preview updates live. Clicking “Apply” commits
the change (with undo/redo support) and adds the function to the pipeline. Undo/redo restore
both the processed image and the pipeline order. The current pipeline is shown beneath the images.
"""

import sys, os, json, logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Dict

import cv2, numpy as np
from skimage import io
from PyQt5 import QtWidgets, QtCore, QtGui

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

#####################################
# 1. CONFIGURATION & UTILITIES
#####################################

class Config:
    SUPPORTED_FORMATS = [".jpg", ".png", ".tiff", ".bmp"]
    OUTPUT_DIR = "output"
    SETTINGS_ORG = "MicroscopicApp"
    SETTINGS_APP = "ImageProcessor"

class Loader:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        _, ext = os.path.splitext(path)
        if ext.lower() not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")
        try:
            image = io.imread(path)  # loaded as RGB
        except Exception as e:
            logging.exception("Failed to load image.")
            raise e
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def parse_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.lower() in ['true', '1']
    return bool(val)

#####################################
# 2. PRE‑PROCESSING FUNCTIONS
#####################################

class Preprocessor:
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        if alpha <= 0:
            raise ValueError("Alpha must be > 0")
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        if gamma <= 0:
            raise ValueError("Gamma must be > 0")
        inv_gamma = 1.0 / gamma
        table = np.array([(i/255.0)**inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def noise_reduction(image: np.ndarray, method: str, ksize: int = 5) -> np.ndarray:
        if ksize % 2 == 0:
            ksize += 1
        if method == "Gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif method == "Median":
            return cv2.medianBlur(image, ksize)
        elif method == "Bilateral":
            return cv2.bilateralFilter(image, ksize, 75, 75)
        else:
            return image

    @staticmethod
    def normalize_intensity(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return cv2.normalize(image, None, alpha, beta, cv2.NORM_MINMAX)

    @staticmethod
    def sharpen(image: np.ndarray, strength: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    @staticmethod
    def select_channel(image: np.ndarray, channel: str) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        B, G, R = cv2.split(image)
        if channel == "All":
            return image
        elif channel == "R":
            return R
        elif channel == "G":
            return G
        elif channel == "B":
            return B
        elif channel == "RG":
            return np.uint8((R.astype(np.float32) + G.astype(np.float32)) / 2)
        elif channel == "GB":
            return np.uint8((G.astype(np.float32) + B.astype(np.float32)) / 2)
        elif channel == "BR":
            return np.uint8((B.astype(np.float32) + R.astype(np.float32)) / 2)
        else:
            return image

    @staticmethod
    def crop_image(image: np.ndarray, x_offset: int, y_offset: int, width: int, height: int, apply_crop: bool = False) -> np.ndarray:
        if not apply_crop:
            overlay = image.copy()
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + width, y_offset + height), (0, 255, 0), thickness=-1)
            alpha = 0.3
            output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            cv2.rectangle(output, (x_offset, y_offset), (x_offset + width, y_offset + height), (0, 255, 0), thickness=2)
            return output
        else:
            return image[y_offset:y_offset+height, x_offset:x_offset+width]

#####################################
# 3. PIPELINE (Extraction‑Style with Order)
#####################################

@dataclass
class PipelineStep:
    name: str
    function: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            logging.debug(f"Skipping disabled step: {self.name}")
            return image
        logging.debug(f"Applying step: {self.name} with params: {self.params}")
        return self.function(image, **self.params)

class PreprocessingPipeline:
    def __init__(self):
        self.steps: List[PipelineStep] = []
    def add_step(self, step: PipelineStep):
        self.steps.append(step)
    def clear_steps(self):
        self.steps = []
    def apply(self, image: np.ndarray) -> np.ndarray:
        processed = image.copy()
        for step in self.steps:
            processed = step.apply(processed)
        return processed

def get_settings_dict(settings: QtCore.QSettings) -> dict:
    d = {}
    for key in settings.allKeys():
        d[key] = settings.value(key)
    return d

def build_preprocessing_pipeline_from_dict(settings_dict: dict) -> PreprocessingPipeline:
    pipeline = PreprocessingPipeline()
    order_str = settings_dict.get("preprocess/order", "")
    order = order_str.split(",") if order_str else []
    for step in order:
        if step == "Grayscale":
            enabled = parse_bool(settings_dict.get("preprocess/grayscale", False))
            func = lambda img: Preprocessor.to_grayscale(img)
            params = {}
        elif step == "BrightnessContrast":
            enabled = parse_bool(settings_dict.get("preprocess/brightness_contrast/enabled", False))
            alpha = float(settings_dict.get("preprocess/brightness_contrast/alpha", 1.0))
            beta = int(settings_dict.get("preprocess/brightness_contrast/beta", 0))
            func = lambda img, alpha, beta: Preprocessor.adjust_contrast_brightness(img, alpha, beta)
            params = {"alpha": alpha, "beta": beta}
        elif step == "Gamma":
            enabled = parse_bool(settings_dict.get("preprocess/gamma/enabled", False))
            gamma_val = float(settings_dict.get("preprocess/gamma/value", 1.0))
            func = lambda img, gamma: Preprocessor.adjust_gamma(img, gamma)
            params = {"gamma": gamma_val}
        elif step == "IntensityNormalization":
            enabled = parse_bool(settings_dict.get("preprocess/normalize/enabled", False))
            norm_alpha = int(settings_dict.get("preprocess/normalize/alpha", 0))
            norm_beta = int(settings_dict.get("preprocess/normalize/beta", 255))
            func = lambda img, alpha, beta: Preprocessor.normalize_intensity(img, alpha, beta)
            params = {"alpha": norm_alpha, "beta": norm_beta}
        elif step == "NoiseReduction":
            enabled = parse_bool(settings_dict.get("preprocess/noise_reduction/enabled", False))
            method = settings_dict.get("preprocess/noise_reduction/method", "Gaussian")
            ksize = int(settings_dict.get("preprocess/noise_reduction/ksize", 5))
            func = lambda img, method, ksize: Preprocessor.noise_reduction(img, method, ksize)
            params = {"method": method, "ksize": ksize}
        elif step == "Sharpen":
            enabled = parse_bool(settings_dict.get("preprocess/sharpen/enabled", False))
            strength = float(settings_dict.get("preprocess/sharpen/strength", 1.0))
            func = lambda img, strength: Preprocessor.sharpen(img, strength)
            params = {"strength": strength}
        elif step == "SelectChannel":
            enabled = parse_bool(settings_dict.get("preprocess/select_channel/enabled", False))
            channel = settings_dict.get("preprocess/select_channel/value", "All")
            func = lambda img, channel: Preprocessor.select_channel(img, channel)
            params = {"channel": channel}
        elif step == "Crop":
            enabled = parse_bool(settings_dict.get("preprocess/crop/enabled", False))
            x_offset = int(settings_dict.get("preprocess/crop/x_offset", 0))
            y_offset = int(settings_dict.get("preprocess/crop/y_offset", 0))
            width = int(settings_dict.get("preprocess/crop/width", 100))
            height = int(settings_dict.get("preprocess/crop/height", 100))
            func = lambda img, x_offset, y_offset, width, height, apply_crop: Preprocessor.crop_image(img, x_offset, y_offset, width, height, apply_crop)
            params = {"x_offset": x_offset, "y_offset": y_offset, "width": width, "height": height, "apply_crop": enabled}
        else:
            continue
        pipeline.add_step(PipelineStep(name=step, function=func, enabled=enabled, params=params))
    return pipeline

def build_preprocessing_pipeline(settings: QtCore.QSettings) -> PreprocessingPipeline:
    return build_preprocessing_pipeline_from_dict(get_settings_dict(settings))

#####################################
# 4. GUI COMPONENTS & DIALOGS
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
        # Convert buffer to bytes
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

# --- Dialogs ---

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
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.setSingleStep(0.1)
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
        form_layout.addRow("Alpha (min):", self.alpha_spin)
        form_layout.addRow("Beta (max):", self.beta_spin)
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
        self.ksize_spin.setRange(1, 31)
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
        self.strength_spin.setRange(1.0, 10.0)
        self.strength_spin.setValue(strength)
        self.strength_spin.setSingleStep(0.5)
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
        form_layout = QtWidgets.QFormLayout()
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(["All", "R", "G", "B", "RG", "GB", "BR"])
        self.channel_combo.setCurrentText(current_channel)
        form_layout.addRow("Channel:", self.channel_combo)
        layout.addLayout(form_layout)
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
    def __init__(self, x_offset: int = 0, y_offset: int = 0, width: int = 100, height: int = 100, preview_callback: Optional[Callable[[int, int, int, int], None]] = None):
        super().__init__()
        self.setWindowTitle("Crop Image")
        self.preview_callback = preview_callback
        self.initial_x = x_offset
        self.initial_y = y_offset
        self.initial_width = width
        self.initial_height = height
        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.x_spin = QtWidgets.QSpinBox()
        self.x_spin.setRange(0, 10000)
        self.x_spin.setValue(x_offset)
        self.y_spin = QtWidgets.QSpinBox()
        self.y_spin.setRange(0, 10000)
        self.y_spin.setValue(y_offset)
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(width)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(1, 10000)
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
            self.preview_callback(self.x_spin.value(), self.y_spin.value(), self.width_spin.value(), self.height_spin.value())
    def get_values(self) -> Tuple[int, int, int, int]:
        return self.x_spin.value(), self.y_spin.value(), self.width_spin.value(), self.height_spin.value()
    def reset_to_initial(self):
        self.x_spin.setValue(self.initial_x)
        self.y_spin.setValue(self.initial_y)
        self.width_spin.setValue(self.initial_width)
        self.height_spin.setValue(self.initial_height)

#####################################
# 5. Helper: Pipeline Order Manager (allows duplicates)
#####################################

class PipelineOrderManager:
    def __init__(self, settings: QtCore.QSettings):
        self.settings = settings
    def get_order(self) -> List[str]:
        order_str = self.settings.value("preprocess/order", "")
        return order_str.split(",") if order_str else []
    def set_order(self, order: List[str]):
        self.settings.setValue("preprocess/order", ",".join(order))
    def append_function(self, func_name: str):
        order = self.get_order()
        order.append(func_name)
        self.set_order(order)

#####################################
# 6. MAIN WINDOW
#####################################

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Pre‑Processing Module")
        self.resize(1200, 700)
        self.original_image: Optional[np.ndarray] = None
        self.processing_image: Optional[np.ndarray] = None
        self.base_image: Optional[np.ndarray] = None
        self.committed_image: Optional[np.ndarray] = None
        self.undo_stack: List[Tuple[np.ndarray, List[str]]] = []
        self.redo_stack: List[Tuple[np.ndarray, List[str]]] = []
        self.current_preview: Optional[np.ndarray] = None
        self.settings = QtCore.QSettings(Config.SETTINGS_ORG, Config.SETTINGS_APP)
        if not self.settings.contains("preprocess/brightness_contrast/alpha"):
            self.settings.setValue("preprocess/brightness_contrast/alpha", 1.0)
        if not self.settings.contains("preprocess/brightness_contrast/beta"):
            self.settings.setValue("preprocess/brightness_contrast/beta", 0)
        if not self.settings.contains("preprocess/gamma/value"):
            self.settings.setValue("preprocess/gamma/value", 1.0)
        if not self.settings.contains("preprocess/select_channel/value"):
            self.settings.setValue("preprocess/select_channel/value", "All")
        self.settings.setValue("preprocess/grayscale", False)
        self.settings.setValue("preprocess/brightness_contrast/enabled", False)
        self.settings.setValue("preprocess/gamma/enabled", False)
        self.settings.setValue("preprocess/hist_eq/enabled", False)
        self.settings.setValue("preprocess/noise_reduction/enabled", False)
        self.settings.setValue("preprocess/sharpen/enabled", False)
        self.settings.setValue("preprocess/select_channel/enabled", False)
        self.settings.setValue("preprocess/crop/enabled", False)
        self.settings.setValue("preprocess/crop/x_offset", 0)
        self.settings.setValue("preprocess/crop/y_offset", 0)
        self.settings.setValue("preprocess/crop/width", 100)
        self.settings.setValue("preprocess/crop/height", 100)
        self.order_manager = PipelineOrderManager(self.settings)
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
        preview_group = QtWidgets.QGroupBox("Pre‑Processing Preview")
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
        self.pipeline = build_preprocessing_pipeline(self.settings)
        self.update_pipeline_label()
        if self.base_image is not None:
            self.update_preview()

    def update_pipeline_label(self):
        order = self.order_manager.get_order()
        self.pipeline_label.setText("Current Pipeline: " + " -> ".join(order) if order else "Current Pipeline: (none)")

    def rebuild_pipeline(self):
        self.pipeline = build_preprocessing_pipeline(self.settings)
        logging.debug("Pipeline rebuilt with steps: " + ", ".join([step.name for step in self.pipeline.steps]))
        self.update_pipeline_label()

    def get_preprocess_order(self) -> List[str]:
        order_str = self.settings.value("preprocess/order", "")
        return order_str.split(",") if order_str else []

    def set_preprocess_order(self, order: List[str]):
        self.settings.setValue("preprocess/order", ",".join(order))
        self.update_pipeline_label()

    def commit_preprocess(self, func_name: str):
        self.order_manager.append_function(func_name)
        self.update_pipeline_label()

    def push_undo_state(self, backup: np.ndarray):
        self.undo_stack.append((backup.copy(), self.get_preprocess_order()))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            current_state = (self.committed_image.copy(), self.get_preprocess_order())
            self.redo_stack.append(current_state)
            prev_image, prev_order = self.undo_stack.pop()
            self.committed_image = prev_image.copy()
            self.set_preprocess_order(prev_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def redo(self):
        if self.redo_stack:
            current_state = (self.committed_image.copy(), self.get_preprocess_order())
            self.undo_stack.append(current_state)
            next_image, next_order = self.redo_stack.pop()
            self.committed_image = next_image.copy()
            self.set_preprocess_order(next_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.redo_stack) > 0)

    def build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_action = QtWidgets.QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        save_action = QtWidgets.QAction("Save Pre-Processed Image", self)
        save_action.triggered.connect(self.save_processed_image)
        file_menu.addAction(save_action)
        mass_pp_action = QtWidgets.QAction("Mass Pre‑Process Folder", self)
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
        reset_action = QtWidgets.QAction("Reset All", self)
        reset_action.triggered.connect(self.reset_all)
        edit_menu.addAction(reset_action)
        pre_menu = menubar.addMenu("Pre‑Processing")
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
        self.settings.setValue("preprocess/grayscale", False)
        self.settings.setValue("preprocess/brightness_contrast/enabled", False)
        self.settings.setValue("preprocess/brightness_contrast/alpha", 1.0)
        self.settings.setValue("preprocess/brightness_contrast/beta", 0)
        self.settings.setValue("preprocess/gamma/enabled", False)
        self.settings.setValue("preprocess/gamma/value", 1.0)
        self.settings.setValue("preprocess/noise_reduction/enabled", False)
        self.settings.setValue("preprocess/sharpen/enabled", False)
        self.settings.setValue("preprocess/select_channel/enabled", False)
        self.settings.setValue("preprocess/select_channel/value", "All")
        self.settings.setValue("preprocess/crop/enabled", False)
        self.settings.setValue("preprocess/crop/x_offset", 0)
        self.settings.setValue("preprocess/crop/y_offset", 0)
        self.settings.setValue("preprocess/crop/width", 100)
        self.settings.setValue("preprocess/crop/height", 100)
        self.order_manager.set_order([])
        self.rebuild_pipeline()
        if self.base_image is not None:
            self.committed_image = self.base_image.copy()
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
            self.statusBar().showMessage("Reset all processing to defaults.")

    def mass_preprocess(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder for Mass Pre‑Process")
        if not folder:
            return
        parent_dir = os.path.dirname(folder)
        base_folder = os.path.basename(folder)
        output_folder = os.path.join(parent_dir, base_folder + "_pp")
        os.makedirs(output_folder, exist_ok=True)
        count = 0
        for file in os.listdir(folder):
            fullpath = os.path.join(folder, file)
            if os.path.isfile(fullpath) and os.path.splitext(file)[1].lower() in Config.SUPPORTED_FORMATS:
                try:
                    image = Loader.load_image(fullpath)
                    processed = self.pipeline.apply(image)
                    name, ext = os.path.splitext(file)
                    new_filename = name + "_pp" + ext
                    outpath = os.path.join(output_folder, new_filename)
                    cv2.imwrite(outpath, processed)
                    count += 1
                except Exception as e:
                    logging.error(f"Failed to process {file}: {e}")
        QtWidgets.QMessageBox.information(self, "Mass Pre‑Process",
                                          f"Processed {count} images.\nOutput folder: {output_folder}")

    def export_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Pipeline Settings", "", "JSON Files (*.json)")
        if filename:
            try:
                pipeline_data = {"order": self.order_manager.get_order()}
                for func in self.order_manager.get_order():
                    if func == "BrightnessContrast":
                        pipeline_data["BrightnessContrast"] = {
                            "alpha": self.settings.value("preprocess/brightness_contrast/alpha"),
                            "beta": self.settings.value("preprocess/brightness_contrast/beta")
                        }
                    elif func == "Gamma":
                        pipeline_data["Gamma"] = {
                            "gamma": self.settings.value("preprocess/gamma/value")
                        }
                    elif func == "IntensityNormalization":
                        pipeline_data["IntensityNormalization"] = {
                            "alpha": self.settings.value("preprocess/normalize/alpha"),
                            "beta": self.settings.value("preprocess/normalize/beta")
                        }
                    elif func == "NoiseReduction":
                        pipeline_data["NoiseReduction"] = {
                            "method": self.settings.value("preprocess/noise_reduction/method"),
                            "ksize": self.settings.value("preprocess/noise_reduction/ksize")
                        }
                    elif func == "Sharpen":
                        pipeline_data["Sharpen"] = {
                            "strength": self.settings.value("preprocess/sharpen/strength")
                        }
                    elif func == "SelectChannel":
                        pipeline_data["SelectChannel"] = {
                            "channel": self.settings.value("preprocess/select_channel/value")
                        }
                    elif func == "Grayscale":
                        pipeline_data["Grayscale"] = {}
                    elif func == "Crop":
                        pipeline_data["Crop"] = {
                            "x_offset": self.settings.value("preprocess/crop/x_offset"),
                            "y_offset": self.settings.value("preprocess/crop/y_offset"),
                            "width": self.settings.value("preprocess/crop/width"),
                            "height": self.settings.value("preprocess/crop/height")
                        }
                with open(filename, 'w') as f:
                    json.dump(pipeline_data, f, indent=2)
                QtWidgets.QMessageBox.information(self, "Pipeline Export", "Pipeline settings exported.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export pipeline: {e}")

    def import_pipeline(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Pipeline Settings", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    pipeline_data = json.load(f)
                order = pipeline_data.get("order", [])
                self.settings.setValue("preprocess/order", ",".join(order))
                if "BrightnessContrast" in pipeline_data:
                    params = pipeline_data["BrightnessContrast"]
                    self.settings.setValue("preprocess/brightness_contrast/alpha", params.get("alpha", 1.0))
                    self.settings.setValue("preprocess/brightness_contrast/beta", params.get("beta", 0))
                    self.settings.setValue("preprocess/brightness_contrast/enabled", True)
                if "Gamma" in pipeline_data:
                    params = pipeline_data["Gamma"]
                    self.settings.setValue("preprocess/gamma/value", params.get("gamma", 1.0))
                    self.settings.setValue("preprocess/gamma/enabled", True)
                if "IntensityNormalization" in pipeline_data:
                    params = pipeline_data["IntensityNormalization"]
                    self.settings.setValue("preprocess/normalize/alpha", params.get("alpha", 0))
                    self.settings.setValue("preprocess/normalize/beta", params.get("beta", 255))
                    self.settings.setValue("preprocess/normalize/enabled", True)
                if "NoiseReduction" in pipeline_data:
                    params = pipeline_data["NoiseReduction"]
                    self.settings.setValue("preprocess/noise_reduction/method", params.get("method", "Gaussian"))
                    self.settings.setValue("preprocess/noise_reduction/ksize", params.get("ksize", 5))
                    self.settings.setValue("preprocess/noise_reduction/enabled", True)
                if "Sharpen" in pipeline_data:
                    params = pipeline_data["Sharpen"]
                    self.settings.setValue("preprocess/sharpen/strength", params.get("strength", 1.0))
                    self.settings.setValue("preprocess/sharpen/enabled", True)
                if "SelectChannel" in pipeline_data:
                    params = pipeline_data["SelectChannel"]
                    self.settings.setValue("preprocess/select_channel/value", params.get("channel", "All"))
                    self.settings.setValue("preprocess/select_channel/enabled", True)
                if "Grayscale" in pipeline_data:
                    self.settings.setValue("preprocess/grayscale", True)
                if "Crop" in pipeline_data:
                    params = pipeline_data["Crop"]
                    self.settings.setValue("preprocess/crop/x_offset", params.get("x_offset", 0))
                    self.settings.setValue("preprocess/crop/y_offset", params.get("y_offset", 0))
                    self.settings.setValue("preprocess/crop/width", params.get("width", 100))
                    self.settings.setValue("preprocess/crop/height", params.get("height", 100))
                    self.settings.setValue("preprocess/crop/enabled", True)
                self.settings.sync()
                self.rebuild_pipeline()
                if self.base_image is not None:
                    self.committed_image = self.pipeline.apply(self.base_image)
                    self.current_preview = self.committed_image.copy()
                    self.preview_display.set_image(self.current_preview)
                QtWidgets.QMessageBox.information(self, "Pipeline Import", "Pipeline settings imported and applied.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to import pipeline: {e}")

    def toggle_grayscale(self):
        backup = self.committed_image.copy() if self.committed_image is not None else None
        current = parse_bool(self.settings.value("preprocess/grayscale", False))
        new_val = not current
        self.settings.setValue("preprocess/grayscale", new_val)
        if new_val:
            self.push_undo_state(backup)
            self.commit_preprocess("Grayscale")
        self.rebuild_pipeline()
        self.committed_image = self.pipeline.apply(self.base_image)
        self.current_preview = self.committed_image.copy()
        self.preview_display.set_image(self.current_preview)

    def preview_update(self, func_name: str, new_params: Dict[str, Any]):
        temp_dict = get_settings_dict(self.settings)
        mapping = {
            "BrightnessContrast": "brightness_contrast",
            "Gamma": "gamma",
            "IntensityNormalization": "normalize",
            "NoiseReduction": "noise_reduction",
            "Sharpen": "sharpen",
            "SelectChannel": "select_channel",
            "Crop": "crop"
        }
        ns = mapping.get(func_name, func_name)
        temp_dict[f"preprocess/{ns}/enabled"] = True
        order = temp_dict.get("preprocess/order", "")
        order_list = order.split(",") if order else []
        temp_order = order_list + [func_name]
        temp_dict["preprocess/order"] = ",".join(temp_order)
        for key, value in new_params.items():
            if func_name == "Gamma" and key == "gamma":
                temp_key = "preprocess/gamma/value"
            else:
                temp_key = f"preprocess/{ns}/{key}"
            temp_dict[temp_key] = value
        if func_name == "Crop":
            temp_dict["preprocess/crop/enabled"] = True
            temp_dict["preprocess/crop/apply_crop"] = False
        temp_pipeline = build_preprocessing_pipeline_from_dict(temp_dict)
        if self.base_image is not None:
            new_preview = temp_pipeline.apply(self.base_image)
            self.current_preview = new_preview.copy()
            self.preview_display.set_image(new_preview)

    def show_brightness_contrast_dialog(self):
        backup = self.committed_image.copy()
        current_alpha = float(self.settings.value("preprocess/brightness_contrast/alpha", 1.0))
        current_beta = int(self.settings.value("preprocess/brightness_contrast/beta", 0))
        dlg = BrightnessContrastDialog(alpha=current_alpha, beta=current_beta,
                                       preview_callback=lambda a, b: self.preview_update("BrightnessContrast", {"alpha": a, "beta": b}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_alpha, new_beta = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_preprocess("BrightnessContrast")
            self.settings.setValue("preprocess/brightness_contrast/enabled", True)
            self.settings.setValue("preprocess/brightness_contrast/alpha", new_alpha)
            self.settings.setValue("preprocess/brightness_contrast/beta", new_beta)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def show_gamma_dialog(self):
        backup = self.committed_image.copy()
        current_gamma = float(self.settings.value("preprocess/gamma/value", 1.0))
        dlg = GammaDialog(gamma=current_gamma, preview_callback=lambda g: self.preview_update("Gamma", {"gamma": g}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_gamma = dlg.get_value()
            self.push_undo_state(backup)
            self.commit_preprocess("Gamma")
            self.settings.setValue("preprocess/gamma/enabled", True)
            self.settings.setValue("preprocess/gamma/value", new_gamma)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def show_normalize_dialog(self):
        backup = self.committed_image.copy()
        current_alpha = int(self.settings.value("preprocess/normalize/alpha", 0))
        current_beta = int(self.settings.value("preprocess/normalize/beta", 255))
        dlg = NormalizeDialog(alpha=current_alpha, beta=current_beta,
                              preview_callback=lambda a, b: self.preview_update("IntensityNormalization", {"alpha": a, "beta": b}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_alpha, new_beta = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_preprocess("IntensityNormalization")
            self.settings.setValue("preprocess/normalize/enabled", True)
            self.settings.setValue("preprocess/normalize/alpha", new_alpha)
            self.settings.setValue("preprocess/normalize/beta", new_beta)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def show_noise_reduction_dialog(self):
        backup = self.committed_image.copy()
        current_method = self.settings.value("preprocess/noise_reduction/method", "Gaussian")
        current_ksize = int(self.settings.value("preprocess/noise_reduction/ksize", 5))
        dlg = NoiseReductionDialog(method=current_method, ksize=current_ksize,
                                   preview_callback=lambda m, k: self.preview_update("NoiseReduction", {"method": m, "ksize": k}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_method, new_ksize = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_preprocess("NoiseReduction")
            self.settings.setValue("preprocess/noise_reduction/enabled", True)
            self.settings.setValue("preprocess/noise_reduction/method", new_method)
            self.settings.setValue("preprocess/noise_reduction/ksize", new_ksize)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def show_sharpen_dialog(self):
        backup = self.committed_image.copy()
        current_strength = float(self.settings.value("preprocess/sharpen/strength", 1.0))
        dlg = SharpenDialog(strength=current_strength, preview_callback=lambda s: self.preview_update("Sharpen", {"strength": s}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_strength = dlg.get_value()
            self.push_undo_state(backup)
            self.commit_preprocess("Sharpen")
            self.settings.setValue("preprocess/sharpen/enabled", True)
            self.settings.setValue("preprocess/sharpen/strength", new_strength)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def show_select_channel_dialog(self):
        backup = self.committed_image.copy()
        current_channel = self.settings.value("preprocess/select_channel/value", "All")
        dlg = SelectChannelDialog(current_channel=current_channel,
                                  preview_callback=lambda c: self.preview_update("SelectChannel", {"channel": c}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_channel = dlg.get_value()
            self.push_undo_state(backup)
            self.commit_preprocess("SelectChannel")
            self.settings.setValue("preprocess/select_channel/enabled", True)
            self.settings.setValue("preprocess/select_channel/value", new_channel)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def show_crop_dialog(self):
        backup = self.committed_image.copy()
        current_x = int(self.settings.value("preprocess/crop/x_offset", 0))
        current_y = int(self.settings.value("preprocess/crop/y_offset", 0))
        current_width = int(self.settings.value("preprocess/crop/width", 100))
        current_height = int(self.settings.value("preprocess/crop/height", 100))
        dlg = CropDialog(x_offset=current_x, y_offset=current_y, width=current_width, height=current_height,
                         preview_callback=lambda x, y, w, h: self.preview_update("Crop", {"x_offset": x, "y_offset": y, "width": w, "height": h, "apply_crop": False}))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_x, new_y, new_w, new_h = dlg.get_values()
            self.push_undo_state(backup)
            self.commit_preprocess("Crop")
            self.settings.setValue("preprocess/crop/enabled", True)
            self.settings.setValue("preprocess/crop/x_offset", new_x)
            self.settings.setValue("preprocess/crop/y_offset", new_y)
            self.settings.setValue("preprocess/crop/width", new_w)
            self.settings.setValue("preprocess/crop/height", new_h)
            # For committed pipeline, set apply_crop to True.
            self.settings.setValue("preprocess/crop/apply_crop", True)
            self.rebuild_pipeline()
            self.committed_image = self.pipeline.apply(self.base_image)
            self.current_preview = self.committed_image.copy()
            self.preview_display.set_image(self.current_preview)
        else:
            self.preview_display.set_image(backup)

    def load_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.bmp *.tiff)")
        if filename:
            try:
                self.original_image = Loader.load_image(filename)
                self.processing_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                self.base_image = self.processing_image.copy()
                self.committed_image = build_preprocessing_pipeline(self.settings).apply(self.base_image)
                self.current_preview = self.committed_image.copy()
                self.original_display.set_image(self.original_image)
                self.preview_display.set_image(self.current_preview)
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.update_undo_redo_actions()
                self.statusBar().showMessage("Image loaded.")
                self.settings.setValue("preprocess/grayscale", False)
                self.settings.setValue("preprocess/brightness_contrast/enabled", False)
                self.settings.setValue("preprocess/gamma/enabled", False)
                self.settings.setValue("preprocess/hist_eq/enabled", False)
                self.settings.setValue("preprocess/noise_reduction/enabled", False)
                self.settings.setValue("preprocess/sharpen/enabled", False)
                self.settings.setValue("preprocess/select_channel/enabled", False)
                self.settings.setValue("preprocess/crop/enabled", False)
                self.order_manager.set_order([])
                self.rebuild_pipeline()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def update_preview(self):
        if self.base_image is None:
            return
        new_preview = self.pipeline.apply(self.base_image)
        self.committed_image = new_preview.copy()
        self.current_preview = new_preview.copy()
        self.preview_display.set_image(new_preview)

    def save_processed_image(self):
        if self.current_preview is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No processed image to save.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Pre-Processed Image", "", "Bitmap Files (*.bmp)")
        if filename:
            try:
                cv2.imwrite(filename, self.current_preview)
                QtWidgets.QMessageBox.information(self, "Save Image", f"Image saved successfully to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.redo_stack) > 0)

    def undo(self):
        if self.undo_stack:
            current_state = (self.committed_image.copy(), self.get_preprocess_order())
            self.redo_stack.append(current_state)
            prev_image, prev_order = self.undo_stack.pop()
            self.committed_image = prev_image.copy()
            self.set_preprocess_order(prev_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

    def redo(self):
        if self.redo_stack:
            current_state = (self.committed_image.copy(), self.get_preprocess_order())
            self.undo_stack.append(current_state)
            next_image, next_order = self.redo_stack.pop()
            self.committed_image = next_image.copy()
            self.set_preprocess_order(next_order)
            self.rebuild_pipeline()
            self.current_preview = self.pipeline.apply(self.base_image)
            self.preview_display.set_image(self.current_preview)
            self.update_undo_redo_actions()

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
