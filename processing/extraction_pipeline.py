"""Pipeline utilities for the extraction workflow."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from PyQt5 import QtCore

from core.extraction import (
    approximate_shape_extraction,
    fourier_descriptors_extraction,
    fractal_dimension_extraction,
    gabor_extraction,
    haralick_extraction,
    histogram_stats_extraction,
    hog_extraction,
    hu_moments_extraction,
    lbp_extraction,
    region_properties_extraction,
)

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

class ProcessingPipeline:
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

#####################################
# 5. SETTINGS FUNCTIONS FOR EXTRACTION PIPELINE
#####################################

def get_extraction_settings_dict(settings: QtCore.QSettings) -> dict:
    d = {}
    for key in settings.allKeys():
        if key.startswith("extraction/"):
            d[key] = settings.value(key)
    return d

def build_extraction_pipeline_from_dict(settings_dict: dict) -> ProcessingPipeline:
    pipeline = ProcessingPipeline()
    order_str = settings_dict.get("extraction/order", "")
    order = order_str.split(",") if order_str else []
    for method in order:
        if method == "Region Properties":
            params = {}
            func = region_properties_extraction
        elif method == "Hu Moments":
            params = {}
            func = hu_moments_extraction
        elif method == "LBP":
            params = {"P": int(settings_dict.get("extraction/LBP/P", 8)),
                      "R": float(settings_dict.get("extraction/LBP/R", 1.0))}
            func = lbp_extraction
        elif method == "Haralick":
            params = {"distance": int(settings_dict.get("extraction/Haralick/distance", 1)),
                      "angle": float(settings_dict.get("extraction/Haralick/angle", 0.0))}
            func = haralick_extraction
        elif method == "Gabor":
            params = {"ksize": int(settings_dict.get("extraction/Gabor/ksize", 21)),
                      "sigma": float(settings_dict.get("extraction/Gabor/sigma", 5.0)),
                      "theta": float(settings_dict.get("extraction/Gabor/theta", 0.0)),
                      "lambd": float(settings_dict.get("extraction/Gabor/lambd", 10.0)),
                      "gamma": float(settings_dict.get("extraction/Gabor/gamma", 0.5)),
                      "psi": float(settings_dict.get("extraction/Gabor/psi", 0.0))}
            func = gabor_extraction
        elif method == "Fourier":
            params = {"num_coeff": int(settings_dict.get("extraction/Fourier/num_coeff", 10))}
            func = fourier_descriptors_extraction
        elif method == "HOG":
            params = {"orientations": int(settings_dict.get("extraction/HOG/orientations", 9)),
                      "pixels_per_cell": (int(settings_dict.get("extraction/HOG/ppc", 8)), int(settings_dict.get("extraction/HOG/ppc", 8))),
                      "cells_per_block": (int(settings_dict.get("extraction/HOG/cpb", 3)), int(settings_dict.get("extraction/HOG/cpb", 3)))}
            func = hog_extraction
        elif method == "Histogram":
            params = {}
            func = histogram_stats_extraction
        elif method == "Fractal":
            params = {"min_box_size": int(settings_dict.get("extraction/Fractal/min_box_size", 2))}
            func = fractal_dimension_extraction
        elif method == "Approximate Shape":
            params = {"error_threshold": float(settings_dict.get("extraction/Approximate Shape/error_threshold", 1.0))}
            func = approximate_shape_extraction
        else:
            continue
        pipeline.add_step(PipelineStep(name=method, function=func, enabled=True, params=params))
    return pipeline

def build_extraction_pipeline(settings: QtCore.QSettings) -> ProcessingPipeline:
    return build_extraction_pipeline_from_dict(get_extraction_settings_dict(settings))

#####################################


__all__ = [
    "PipelineStep",
    "ProcessingPipeline",
    "build_extraction_pipeline",
    "build_extraction_pipeline_from_dict",
    "get_extraction_settings_dict",
]
