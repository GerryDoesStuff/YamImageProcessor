"""Pipeline utilities for the preprocessing tool."""
from __future__ import annotations

from typing import Dict, Optional

from core.app_core import AppCore
from core.preprocessing import Preprocessor, parse_bool
from yam_processor.processing import PipelineManager, PipelineStep


class PreprocessingPipeline(PipelineManager):
    """Thin wrapper retaining the historical name used in the UI code."""

    def __init__(self, app_core: Optional[AppCore] = None) -> None:
        super().__init__()
        self.app_core = app_core
        self.thread_controller = getattr(app_core, "thread_controller", None)


def get_settings_dict(app_core: AppCore) -> Dict[str, object]:
    """Extract settings from :class:`AppCore` into a plain dictionary."""

    settings = app_core.qsettings
    return {key: settings.value(key) for key in settings.allKeys()}


def build_preprocessing_pipeline_from_dict(
    settings_dict: Dict[str, object],
    app_core: Optional[AppCore] = None,
) -> PreprocessingPipeline:
    pipeline = PreprocessingPipeline(app_core)
    order_str = settings_dict.get("preprocess/order", "") or ""
    order = order_str.split(",") if order_str else []

    for step in order:
        if step == "Grayscale":
            enabled = parse_bool(settings_dict.get("preprocess/grayscale", False))
            params = {}
            func = Preprocessor.to_grayscale
        elif step == "BrightnessContrast":
            enabled = parse_bool(settings_dict.get("preprocess/brightness_contrast/enabled", False))
            alpha = float(settings_dict.get("preprocess/brightness_contrast/alpha", 1.0))
            beta = int(settings_dict.get("preprocess/brightness_contrast/beta", 0))
            params = {"alpha": alpha, "beta": beta}
            func = Preprocessor.adjust_contrast_brightness
        elif step == "Gamma":
            enabled = parse_bool(settings_dict.get("preprocess/gamma/enabled", False))
            gamma_val = float(settings_dict.get("preprocess/gamma/value", 1.0))
            params = {"gamma": gamma_val}
            func = Preprocessor.adjust_gamma
        elif step == "IntensityNormalization":
            enabled = parse_bool(settings_dict.get("preprocess/normalize/enabled", False))
            norm_alpha = int(settings_dict.get("preprocess/normalize/alpha", 0))
            norm_beta = int(settings_dict.get("preprocess/normalize/beta", 255))
            params = {"alpha": norm_alpha, "beta": norm_beta}
            func = Preprocessor.normalize_intensity
        elif step == "NoiseReduction":
            enabled = parse_bool(settings_dict.get("preprocess/noise_reduction/enabled", False))
            method = settings_dict.get("preprocess/noise_reduction/method", "Gaussian")
            ksize = int(settings_dict.get("preprocess/noise_reduction/ksize", 5))
            params = {"method": method, "ksize": ksize}
            func = Preprocessor.noise_reduction
        elif step == "Sharpen":
            enabled = parse_bool(settings_dict.get("preprocess/sharpen/enabled", False))
            strength = float(settings_dict.get("preprocess/sharpen/strength", 1.0))
            params = {"strength": strength}
            func = Preprocessor.sharpen
        elif step == "SelectChannel":
            enabled = parse_bool(settings_dict.get("preprocess/select_channel/enabled", False))
            channel = settings_dict.get("preprocess/select_channel/value", "All")
            params = {"channel": channel}
            func = Preprocessor.select_channel
        elif step == "Crop":
            enabled = parse_bool(settings_dict.get("preprocess/crop/enabled", False))
            x_offset = int(settings_dict.get("preprocess/crop/x_offset", 0))
            y_offset = int(settings_dict.get("preprocess/crop/y_offset", 0))
            width = int(settings_dict.get("preprocess/crop/width", 100))
            height = int(settings_dict.get("preprocess/crop/height", 100))
            params = {
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": width,
                "height": height,
                "apply_crop": enabled,
            }
            func = Preprocessor.crop_image
        else:
            continue

        pipeline.add_step(
            PipelineStep(name=step, function=func, enabled=enabled, params=params)
        )

    return pipeline


def build_preprocessing_pipeline(app_core: AppCore) -> PreprocessingPipeline:
    return build_preprocessing_pipeline_from_dict(get_settings_dict(app_core), app_core)


__all__ = [
    "PreprocessingPipeline",
    "build_preprocessing_pipeline",
    "build_preprocessing_pipeline_from_dict",
    "get_settings_dict",
]
