"""Core logic for the image preprocessing application."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import cv2
import numpy as np
from skimage import io


class Config:
    """Application configuration constants."""

    SUPPORTED_FORMATS = [".jpg", ".png", ".tiff", ".bmp", ".npy"]
    OUTPUT_DIR = "output"
    SETTINGS_ORG = "MicroscopicApp"
    SETTINGS_APP = "ImageProcessor"


class Loader:
    """Utility helpers for loading data from disk."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")
        if ext == ".npy":
            return np.load(path, allow_pickle=False)
        try:
            image = io.imread(path)  # loaded as RGB
        except Exception as exc:  # pragma: no cover - passthrough logging
            logging.exception("Failed to load image.")
            raise exc
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def parse_bool(val: Any) -> bool:
    """Interpret typical truthy values coming from QSettings."""

    if isinstance(val, str):
        return val.lower() in ["true", "1"]
    return bool(val)


class Preprocessor:
    """Collection of image preprocessing kernels."""

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
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def noise_reduction(image: np.ndarray, method: str, ksize: int = 5) -> np.ndarray:
        if ksize % 2 == 0:
            ksize += 1
        if method == "Gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        if method == "Median":
            return cv2.medianBlur(image, ksize)
        if method == "Bilateral":
            return cv2.bilateralFilter(image, ksize, 75, 75)
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
        blue, green, red = cv2.split(image)
        if channel == "All":
            return image
        if channel == "R":
            return red
        if channel == "G":
            return green
        if channel == "B":
            return blue
        if channel == "RG":
            return np.uint8((red.astype(np.float32) + green.astype(np.float32)) / 2)
        if channel == "GB":
            return np.uint8((green.astype(np.float32) + blue.astype(np.float32)) / 2)
        if channel == "BR":
            return np.uint8((blue.astype(np.float32) + red.astype(np.float32)) / 2)
        return image

    @staticmethod
    def crop_image(
        image: np.ndarray,
        x_offset: int,
        y_offset: int,
        width: int,
        height: int,
        apply_crop: bool = False,
    ) -> np.ndarray:
        if not apply_crop:
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (x_offset, y_offset),
                (x_offset + width, y_offset + height),
                (0, 255, 0),
                thickness=-1,
            )
            alpha = 0.3
            output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            cv2.rectangle(
                output,
                (x_offset, y_offset),
                (x_offset + width, y_offset + height),
                (0, 255, 0),
                thickness=2,
            )
            return output
        return image[y_offset : y_offset + height, x_offset : x_offset + width]


__all__ = [
    "Config",
    "Loader",
    "parse_bool",
    "Preprocessor",
]
