"""Pre-processing plugin implementations."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, Protocol, cast

import cv2
import numpy as np

from plugins.module_base import ModuleBase, ModuleMetadata, ModuleStage


if TYPE_CHECKING:
    from ui import ModulePane

    class _PreprocessingPane(Protocol):
        """Structural typing helper describing preprocessing pane affordances."""

        def toggle_grayscale(self) -> None: ...

        def show_brightness_contrast_dialog(self) -> None: ...

        def show_gamma_dialog(self) -> None: ...

        def show_normalize_dialog(self) -> None: ...

        def show_noise_reduction_dialog(self) -> None: ...

        def show_sharpen_dialog(self) -> None: ...

        def show_select_channel_dialog(self) -> None: ...

        def show_crop_dialog(self) -> None: ...


def _preprocessing_pane(pane: "ModulePane") -> "_PreprocessingPane":
    """Cast ``pane`` to the richer preprocessing pane protocol for typing."""

    return cast("_PreprocessingPane", pane)


class GrayscaleModule(ModuleBase):
    """Convert images to grayscale when enabled."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="Grayscale",
            title="Toggle Greyscale",
            stage=ModuleStage.PREPROCESSING,
            description="Toggle grayscale conversion of the current image.",
        )

    def process(self, image: np.ndarray, **_: Any) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).toggle_grayscale()


class BrightnessContrastModule(ModuleBase):
    """Adjust image brightness and contrast."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="BrightnessContrast",
            title="Brightness / Contrast",
            stage=ModuleStage.PREPROCESSING,
            description="Adjust overall brightness and contrast levels.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 0))
        if alpha <= 0:
            raise ValueError("Alpha must be > 0")
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_brightness_contrast_dialog()


class GammaCorrectionModule(ModuleBase):
    """Apply gamma correction."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="Gamma",
            title="Gamma Correction",
            stage=ModuleStage.PREPROCESSING,
            description="Apply gamma correction to the image intensities.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        gamma = float(params.get("gamma", 1.0))
        if gamma <= 0:
            raise ValueError("Gamma must be > 0")
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_gamma_dialog()


class IntensityNormalizationModule(ModuleBase):
    """Normalise intensities to a configured range."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="IntensityNormalization",
            title="Intensity Normalization",
            stage=ModuleStage.PREPROCESSING,
            description="Stretch intensities into the specified range.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        alpha = float(params.get("alpha", 0))
        beta = float(params.get("beta", 255))
        return cv2.normalize(image, None, alpha, beta, cv2.NORM_MINMAX)

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_normalize_dialog()


class NoiseReductionModule(ModuleBase):
    """Apply configurable noise reduction filters."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="NoiseReduction",
            title="Noise Reduction",
            stage=ModuleStage.PREPROCESSING,
            description="Apply Gaussian, median, or bilateral denoising.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        method = str(params.get("method", "Gaussian"))
        ksize = int(params.get("ksize", 5))
        if method == "Gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        if method == "Median":
            return cv2.medianBlur(image, ksize)
        if method == "Bilateral":
            return cv2.bilateralFilter(image, ksize, 75, 75)
        return image

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_noise_reduction_dialog()


class SharpenModule(ModuleBase):
    """Sharpen the image using an unsharp mask."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="Sharpen",
            title="Sharpen",
            stage=ModuleStage.PREPROCESSING,
            description="Enhance edges using a configurable strength.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        strength = float(params.get("strength", 1.0))
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_sharpen_dialog()


class SelectChannelModule(ModuleBase):
    """Select or combine colour channels."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="SelectChannel",
            title="Select Color Channel",
            stage=ModuleStage.PREPROCESSING,
            description="Extract or combine specific colour channels.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        channel = str(params.get("channel", "All"))
        working = image
        if working.ndim == 2:
            working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        blue, green, red = cv2.split(working)
        if channel == "All":
            return working
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
        return working

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_select_channel_dialog()


class CropModule(ModuleBase):
    """Crop the image or preview a crop overlay."""

    def _build_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            identifier="Crop",
            title="Crop",
            stage=ModuleStage.PREPROCESSING,
            description="Crop to a region of interest or preview the crop.",
        )

    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        params = self.sanitize_parameters(kwargs)
        x_offset = int(params.get("x_offset", 0))
        y_offset = int(params.get("y_offset", 0))
        width = int(params.get("width", 100))
        height = int(params.get("height", 100))
        apply_crop = bool(params.get("apply_crop", False))
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

    def activate(self, pane: "ModulePane") -> None:
        _preprocessing_pane(pane).show_crop_dialog()


MODULE_CLASSES = (
    GrayscaleModule,
    BrightnessContrastModule,
    GammaCorrectionModule,
    IntensityNormalizationModule,
    NoiseReductionModule,
    SharpenModule,
    SelectChannelModule,
    CropModule,
)


def register_module(app_core) -> None:
    """Register all preprocessing module classes with ``app_core``."""

    for module_cls in MODULE_CLASSES:
        app_core.register_module(module_cls)


__all__ = [module_cls.__name__ for module_cls in MODULE_CLASSES] + ["register_module"]
