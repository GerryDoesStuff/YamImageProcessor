"""Pipeline builders for the segmentation workflow."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from PyQt5 import QtCore

from core.segmentation import (
    Preprocessor,
    active_contour,
    fuzzy_c_means,
    gmm_segmentation,
    global_threshold,
    graph_cuts,
    laplacian_operator,
    mean_shift_segmentation,
    morphological_closing,
    morphological_dilation,
    morphological_erosion,
    morphological_opening,
    otsu_threshold,
    prewitt_operator,
    region_growing,
    region_splitting_merging,
    remove_border_regions,
    sobel_operator,
)
from core.segmentation import Detector  # type: ignore  # Detector defined alongside Preprocessor

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

def get_settings_dict(settings: QtCore.QSettings) -> dict:
    d = {}
    for key in settings.allKeys():
        d[key] = settings.value(key)
    return d

def build_segmentation_pipeline_from_dict(settings_dict: dict) -> ProcessingPipeline:
    pipeline = ProcessingPipeline()
    order_str = settings_dict.get("segmentation/order", "")
    order = order_str.split(",") if order_str else []
    for method in order:
        if method == "Global":
            threshold = int(settings_dict.get("segmentation/Global/threshold", 127))
            params = {"threshold": threshold}
            func = global_threshold
        elif method == "Otsu":
            params = {}
            func = otsu_threshold
        elif method == "Adaptive":
            params = {"block_size": int(settings_dict.get("segmentation/Adaptive/block_size", 11)),
                      "C": int(settings_dict.get("segmentation/Adaptive/C", 2))}
            func = Detector.adaptive_threshold
        elif method == "Edge":
            params = {"low_threshold": int(settings_dict.get("segmentation/Edge/low_threshold", 50)),
                      "high_threshold": int(settings_dict.get("segmentation/Edge/high_threshold", 150)),
                      "aperture_size": int(settings_dict.get("segmentation/Edge/aperture_size", 3))}
            func = Detector.edge_based_segmentation
        elif method == "Watershed":
            params = {"kernel_size": int(settings_dict.get("segmentation/Watershed/kernel_size", 3)),
                      "opening_iterations": int(settings_dict.get("segmentation/Watershed/opening_iterations", 2)),
                      "dilation_iterations": int(settings_dict.get("segmentation/Watershed/dilation_iterations", 3)),
                      "distance_threshold_factor": float(settings_dict.get("segmentation/Watershed/distance_threshold_factor", 0.7))}
            func = Detector.watershed_segmentation
        elif method == "Sobel":
            params = {"ksize": int(settings_dict.get("segmentation/Sobel/ksize", 3))}
            func = sobel_operator
        elif method == "Prewitt":
            params = {}
            func = prewitt_operator
        elif method == "Laplacian":
            params = {"ksize": int(settings_dict.get("segmentation/Laplacian/ksize", 3))}
            func = laplacian_operator
        elif method == "Region Growing":
            seed = (int(settings_dict.get("segmentation/Region Growing/seed_x", 50)),
                    int(settings_dict.get("segmentation/Region Growing/seed_y", 50)))
            tol = int(settings_dict.get("segmentation/Region Growing/tolerance", 10))
            params = {"seed": seed, "tolerance": tol}
            func = region_growing
        elif method == "Region Splitting/Merging":
            params = {"min_size": int(settings_dict.get("segmentation/Region Splitting/Merging/min_size", 16)),
                      "std_thresh": float(settings_dict.get("segmentation/Region Splitting/Merging/std_thresh", 10))}
            func = region_splitting_merging
        elif method == "K-Means":
            params = {"K": int(settings_dict.get("segmentation/K-Means/K", 2)),
                      "seed": int(settings_dict.get("segmentation/K-Means/seed", 42))}
            func = Detector.kmeans_segmentation
        elif method == "Fuzzy C-Means":
            params = {"K": int(settings_dict.get("segmentation/Fuzzy C-Means/K", 2)),
                      "seed": int(settings_dict.get("segmentation/Fuzzy C-Means/seed", 42))}
            func = fuzzy_c_means
        elif method == "Mean Shift":
            params = {"spatial_radius": int(settings_dict.get("segmentation/Mean Shift/spatial_radius", 20)),
                      "color_radius": int(settings_dict.get("segmentation/Mean Shift/color_radius", 30))}
            func = mean_shift_segmentation
        elif method == "GMM":
            params = {"components": int(settings_dict.get("segmentation/GMM/components", 2)),
                      "seed": int(settings_dict.get("segmentation/GMM/seed", 42))}
            func = gmm_segmentation
        elif method == "Graph Cuts":
            params = {}
            func = graph_cuts
        elif method == "Active Contour":
            params = {"iterations": int(settings_dict.get("segmentation/Active Contour/iterations", 250)),
                      "alpha": float(settings_dict.get("segmentation/Active Contour/alpha", 0.015)),
                      "beta": float(settings_dict.get("segmentation/Active Contour/beta", 10)),
                      "gamma": float(settings_dict.get("segmentation/Active Contour/gamma", 0.001))}
            func = active_contour
        elif method == "Opening":
            params = {"kernel_shape": str(settings_dict.get("segmentation/Opening/kernel_shape", "Rectangular")),
                      "kernel_size": int(settings_dict.get("segmentation/Opening/kernel_size", 3)),
                      "iterations": int(settings_dict.get("segmentation/Opening/iterations", 1))}
            func = morphological_opening
        elif method == "Closing":
            params = {"kernel_shape": str(settings_dict.get("segmentation/Closing/kernel_shape", "Rectangular")),
                      "kernel_size": int(settings_dict.get("segmentation/Closing/kernel_size", 3)),
                      "iterations": int(settings_dict.get("segmentation/Closing/iterations", 1))}
            func = morphological_closing
        elif method == "Dilation":
            params = {"kernel_shape": str(settings_dict.get("segmentation/Dilation/kernel_shape", "Rectangular")),
                      "kernel_size": int(settings_dict.get("segmentation/Dilation/kernel_size", 3)),
                      "iterations": int(settings_dict.get("segmentation/Dilation/iterations", 1))}
            func = morphological_dilation
        elif method == "Erosion":
            params = {"kernel_shape": str(settings_dict.get("segmentation/Erosion/kernel_shape", "Rectangular")),
                      "kernel_size": int(settings_dict.get("segmentation/Erosion/kernel_size", 3)),
                      "iterations": int(settings_dict.get("segmentation/Erosion/iterations", 1))}
            func = morphological_erosion
        elif method == "Border Removal":
            params = {"border_distance": int(settings_dict.get("segmentation/Border Removal/border_distance", 100))}
            func = remove_border_regions
        else:
            continue
        pipeline.add_step(PipelineStep(name=method, function=func, enabled=True, params=params))
    return pipeline

def build_segmentation_pipeline(settings: QtCore.QSettings) -> ProcessingPipeline:
    return build_segmentation_pipeline_from_dict(get_settings_dict(settings))


__all__ = [
    "PipelineStep",
    "ProcessingPipeline",
    "build_segmentation_pipeline",
    "build_segmentation_pipeline_from_dict",
    "get_settings_dict",
]
