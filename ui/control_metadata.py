from __future__ import annotations

"""Centralised metadata for UI controls and module parameters."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


class ControlValueType(Enum):
    """Supported primitive types for control metadata."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"


@dataclass(frozen=True)
class ChoiceOption:
    """Describes an individual selectable option for a combo box control."""

    label: str
    value: Any
    description: str = ""


@dataclass(frozen=True)
class ControlMetadata:
    """Declarative metadata for a single module parameter/control."""

    description: str
    default: Any | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int | None = None
    tooltip: str | None = None
    value_type: ControlValueType | None = None
    choices: tuple[ChoiceOption, ...] | None = None
    coerce_fn: Callable[[Any], Any] | None = None

    def tooltip_text(self) -> str:
        """Return a formatted tooltip string for the control."""

        if self.tooltip:
            return self.tooltip

        parts: list[str] = []
        if self.description:
            parts.append(self.description)

        range_bits: list[str] = []
        if self.minimum is not None and self.maximum is not None:
            range_bits.append(f"Range: {self.minimum}–{self.maximum}")
        elif self.minimum is not None:
            range_bits.append(f"Minimum: {self.minimum}")
        elif self.maximum is not None:
            range_bits.append(f"Maximum: {self.maximum}")

        if self.step is not None and self.value_type in {ControlValueType.FLOAT, ControlValueType.INTEGER}:
            range_bits.append(f"Step: {self.step}")

        if range_bits:
            parts.append("; ".join(range_bits))

        if self.default not in (None, ""):
            parts.append(f"Default: {self.default}")

        if self.choices:
            choice_text = "; ".join(
                option.label if not option.description else f"{option.label} ({option.description})"
                for option in self.choices
            )
            if choice_text:
                parts.append(f"Choices: {choice_text}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Validation helpers
    def _converter(self) -> Callable[[Any], Any] | None:
        if self.value_type is ControlValueType.INTEGER:
            return lambda value: int(round(float(value)))
        if self.value_type is ControlValueType.FLOAT:
            return float
        if self.value_type is ControlValueType.BOOLEAN:
            return lambda value: bool(value)
        if self.value_type is ControlValueType.STRING:
            return lambda value: str(value)
        if self.default is not None:
            return lambda value: type(self.default)(value)  # type: ignore[misc]
        return None

    def coerce(self, value: Any | None) -> Any | None:
        """Coerce ``value`` into a valid representation for the control."""

        if value is None:
            return self.default

        converter = self._converter()
        try:
            if converter is not None:
                value = converter(value)
        except (TypeError, ValueError):
            return self.default

        if self.coerce_fn is not None:
            value = self.coerce_fn(value)

        if isinstance(value, (int, float)):
            if self.minimum is not None:
                value = max(value, self.minimum)
            if self.maximum is not None:
                value = min(value, self.maximum)
            if self.value_type is ControlValueType.INTEGER:
                value = int(round(value))
            elif self.value_type is ControlValueType.FLOAT and self.decimals is not None:
                value = round(float(value), self.decimals)

        if self.choices:
            allowed_values = {option.value for option in self.choices}
            if value not in allowed_values:
                return self.default if self.default is not None else next(iter(allowed_values), self.default)

        return value


# ----------------------------------------------------------------------
# Registry definition helpers

def _ensure_odd(value: Any) -> Any:
    numeric = int(round(float(value)))
    if numeric % 2 == 0:
        numeric += 1
    return numeric


def _choice_options(options: Sequence[tuple[str, Any, str | None]]) -> tuple[ChoiceOption, ...]:
    return tuple(
        ChoiceOption(label=label, value=value, description=description or "")
        for label, value, description in options
    )


MODULE_CONTROL_METADATA: Dict[str, Dict[str, ControlMetadata]] = {
    # ------------------------------------------------------------------
    # Pre-processing modules
    "BrightnessContrast": {
        "alpha": ControlMetadata(
            description="Scale factor that amplifies or reduces overall contrast.",
            default=1.0,
            minimum=0.1,
            maximum=3.0,
            step=0.1,
            decimals=2,
            value_type=ControlValueType.FLOAT,
        ),
        "beta": ControlMetadata(
            description="Additive offset that brightens or darkens the image.",
            default=0,
            minimum=-100,
            maximum=100,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "Gamma": {
        "gamma": ControlMetadata(
            description="Exponent used to remap intensities for gamma correction.",
            default=1.0,
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            decimals=2,
            value_type=ControlValueType.FLOAT,
        ),
    },
    "IntensityNormalization": {
        "alpha": ControlMetadata(
            description="Lower bound for the remapped intensity range.",
            default=0,
            minimum=0,
            maximum=255,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "beta": ControlMetadata(
            description="Upper bound for the remapped intensity range.",
            default=255,
            minimum=0,
            maximum=255,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "NoiseReduction": {
        "method": ControlMetadata(
            description="Select the smoothing approach to suppress noise while preserving detail.",
            default="Gaussian",
            choices=_choice_options(
                [
                    ("Gaussian", "Gaussian", "Fast blur that preserves global structure."),
                    ("Median", "Median", "Effective for salt-and-pepper noise."),
                    ("Bilateral", "Bilateral", "Edge-aware smoothing that retains edges."),
                ]
            ),
            value_type=ControlValueType.STRING,
        ),
        "ksize": ControlMetadata(
            description="Kernel size controlling the neighbourhood used by the filter.",
            default=5,
            minimum=1,
            maximum=15,
            step=2,
            value_type=ControlValueType.INTEGER,
            coerce_fn=_ensure_odd,
        ),
    },
    "Sharpen": {
        "strength": ControlMetadata(
            description="Weight applied to the unsharp mask to emphasise edges.",
            default=1.0,
            minimum=0.0,
            maximum=5.0,
            step=0.1,
            decimals=2,
            value_type=ControlValueType.FLOAT,
        ),
    },
    "SelectChannel": {
        "channel": ControlMetadata(
            description="Colour channel or combination to extract from the source image.",
            default="All",
            choices=_choice_options(
                [
                    ("All", "All", "Leave the image unchanged."),
                    ("R", "R", "Use only the red channel."),
                    ("G", "G", "Use only the green channel."),
                    ("B", "B", "Use only the blue channel."),
                    ("RG", "RG", "Average red and green channels."),
                    ("GB", "GB", "Average green and blue channels."),
                    ("BR", "BR", "Average blue and red channels."),
                ]
            ),
            value_type=ControlValueType.STRING,
        ),
    },
    "Crop": {
        "x_offset": ControlMetadata(
            description="Horizontal offset of the crop window in pixels.",
            default=0,
            minimum=0,
            maximum=5000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "y_offset": ControlMetadata(
            description="Vertical offset of the crop window in pixels.",
            default=0,
            minimum=0,
            maximum=5000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "width": ControlMetadata(
            description="Width of the crop window in pixels.",
            default=100,
            minimum=1,
            maximum=5000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "height": ControlMetadata(
            description="Height of the crop window in pixels.",
            default=100,
            minimum=1,
            maximum=5000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "apply_crop": ControlMetadata(
            description="When enabled the crop is permanently applied instead of previewing an overlay.",
            default=False,
            value_type=ControlValueType.BOOLEAN,
        ),
    },
    # ------------------------------------------------------------------
    # Segmentation modules
    "GlobalThreshold": {
        "threshold": ControlMetadata(
            description="Binary threshold separating foreground from background.",
            default=127,
            minimum=0,
            maximum=255,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "AdaptiveThreshold": {
        "block_size": ControlMetadata(
            description="Neighbourhood size used to compute local thresholds (odd numbers only).",
            default=11,
            minimum=3,
            maximum=101,
            step=2,
            value_type=ControlValueType.INTEGER,
            coerce_fn=_ensure_odd,
        ),
        "C": ControlMetadata(
            description="Constant subtracted from the computed local mean.",
            default=2,
            minimum=-10,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "EdgeBasedSegmentation": {
        "low_threshold": ControlMetadata(
            description="Lower hysteresis threshold for Canny edge detection.",
            default=50,
            minimum=0,
            maximum=255,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "high_threshold": ControlMetadata(
            description="Upper hysteresis threshold for Canny edge detection.",
            default=150,
            minimum=0,
            maximum=255,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "aperture_size": ControlMetadata(
            description="Size of the Sobel kernel used for gradient computation (odd values).",
            default=3,
            minimum=3,
            maximum=7,
            step=2,
            value_type=ControlValueType.INTEGER,
            coerce_fn=_ensure_odd,
        ),
    },
    "Watershed": {
        "kernel_size": ControlMetadata(
            description="Structuring element size for morphological pre-processing.",
            default=3,
            minimum=1,
            maximum=15,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "opening_iterations": ControlMetadata(
            description="Number of opening iterations applied before distance transform.",
            default=2,
            minimum=1,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "dilation_iterations": ControlMetadata(
            description="Number of dilation iterations applied to expand markers.",
            default=3,
            minimum=1,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "distance_threshold_factor": ControlMetadata(
            description="Fraction of the maximum distance used to select sure foreground regions.",
            default=0.7,
            minimum=0.1,
            maximum=1.0,
            step=0.05,
            decimals=2,
            value_type=ControlValueType.FLOAT,
        ),
    },
    "Sobel": {
        "ksize": ControlMetadata(
            description="Kernel size for the Sobel operator (odd values).",
            default=3,
            minimum=1,
            maximum=31,
            step=2,
            value_type=ControlValueType.INTEGER,
            coerce_fn=_ensure_odd,
        ),
    },
    "Laplacian": {
        "ksize": ControlMetadata(
            description="Kernel size for the Laplacian operator (odd values).",
            default=3,
            minimum=1,
            maximum=31,
            step=2,
            value_type=ControlValueType.INTEGER,
            coerce_fn=_ensure_odd,
        ),
    },
    "RegionGrowing": {
        "seed_x": ControlMetadata(
            description="X coordinate of the seed point to start region growing.",
            default=50,
            minimum=0,
            maximum=1000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "seed_y": ControlMetadata(
            description="Y coordinate of the seed point to start region growing.",
            default=50,
            minimum=0,
            maximum=1000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "tolerance": ControlMetadata(
            description="Maximum intensity difference allowed when expanding the region.",
            default=10,
            minimum=0,
            maximum=100,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "RegionSplittingMerging": {
        "min_size": ControlMetadata(
            description="Minimum region size retained during splitting and merging.",
            default=16,
            minimum=1,
            maximum=1000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "std_thresh": ControlMetadata(
            description="Threshold on region standard deviation used for merging decisions.",
            default=10.0,
            minimum=0.1,
            maximum=100.0,
            step=0.5,
            decimals=2,
            value_type=ControlValueType.FLOAT,
        ),
    },
    "KMeans": {
        "K": ControlMetadata(
            description="Number of clusters for k-means segmentation.",
            default=2,
            minimum=2,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "seed": ControlMetadata(
            description="Random seed to ensure reproducible clustering.",
            default=42,
            minimum=0,
            maximum=10000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "FuzzyCMeans": {
        "K": ControlMetadata(
            description="Number of fuzzy clusters to extract.",
            default=2,
            minimum=2,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "seed": ControlMetadata(
            description="Random seed for consistent fuzzy clustering initialisation.",
            default=42,
            minimum=0,
            maximum=10000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "MeanShift": {
        "spatial_radius": ControlMetadata(
            description="Bandwidth controlling spatial proximity in mean shift.",
            default=20,
            minimum=1,
            maximum=100,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "color_radius": ControlMetadata(
            description="Bandwidth controlling colour similarity in mean shift.",
            default=30,
            minimum=1,
            maximum=100,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "GMM": {
        "components": ControlMetadata(
            description="Number of Gaussian components to fit.",
            default=2,
            minimum=2,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "seed": ControlMetadata(
            description="Random seed for reproducible expectation-maximisation.",
            default=42,
            minimum=0,
            maximum=10000,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "ActiveContour": {
        "iterations": ControlMetadata(
            description="Maximum number of snake optimisation iterations.",
            default=250,
            minimum=50,
            maximum=1000,
            step=10,
            value_type=ControlValueType.INTEGER,
        ),
        "alpha": ControlMetadata(
            description="Snake tension parameter controlling elasticity.",
            default=0.015,
            minimum=0.001,
            maximum=0.1,
            step=0.001,
            decimals=3,
            value_type=ControlValueType.FLOAT,
        ),
        "beta": ControlMetadata(
            description="Snake rigidity parameter controlling smoothness.",
            default=10.0,
            minimum=1.0,
            maximum=20.0,
            step=0.5,
            decimals=2,
            value_type=ControlValueType.FLOAT,
        ),
        "gamma": ControlMetadata(
            description="Gradient descent step size for contour evolution.",
            default=0.001,
            minimum=0.0001,
            maximum=0.01,
            step=0.0001,
            decimals=4,
            value_type=ControlValueType.FLOAT,
        ),
    },
    "Opening": {
        "kernel_shape": ControlMetadata(
            description="Structuring element shape used for the morphological operation.",
            default="Rectangular",
            choices=_choice_options(
                [
                    ("Rectangular", "Rectangular", "Standard axis-aligned kernel."),
                    ("Elliptical", "Elliptical", "Softer kernel suited to rounded structures."),
                    ("Cross", "Cross", "Thin kernel ideal for preserving thin features."),
                ]
            ),
            value_type=ControlValueType.STRING,
        ),
        "kernel_size": ControlMetadata(
            description="Size of the structuring element in pixels.",
            default=3,
            minimum=1,
            maximum=31,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "iterations": ControlMetadata(
            description="Number of times the operation is applied sequentially.",
            default=1,
            minimum=1,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "Closing": {
        "kernel_shape": ControlMetadata(
            description="Structuring element shape used for morphological closing.",
            default="Rectangular",
            choices=_choice_options(
                [
                    ("Rectangular", "Rectangular", "Standard axis-aligned kernel."),
                    ("Elliptical", "Elliptical", "Softer kernel suited to rounded structures."),
                    ("Cross", "Cross", "Thin kernel ideal for preserving thin features."),
                ]
            ),
            value_type=ControlValueType.STRING,
        ),
        "kernel_size": ControlMetadata(
            description="Size of the structuring element in pixels.",
            default=3,
            minimum=1,
            maximum=31,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "iterations": ControlMetadata(
            description="Number of closing passes performed.",
            default=1,
            minimum=1,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "Dilation": {
        "kernel_shape": ControlMetadata(
            description="Structuring element shape used for dilation.",
            default="Rectangular",
            choices=_choice_options(
                [
                    ("Rectangular", "Rectangular", "Standard axis-aligned kernel."),
                    ("Elliptical", "Elliptical", "Softer kernel suited to rounded structures."),
                    ("Cross", "Cross", "Thin kernel ideal for preserving thin features."),
                ]
            ),
            value_type=ControlValueType.STRING,
        ),
        "kernel_size": ControlMetadata(
            description="Size of the structuring element in pixels.",
            default=3,
            minimum=1,
            maximum=31,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "iterations": ControlMetadata(
            description="Number of dilation passes performed.",
            default=1,
            minimum=1,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "Erosion": {
        "kernel_shape": ControlMetadata(
            description="Structuring element shape used for erosion.",
            default="Rectangular",
            choices=_choice_options(
                [
                    ("Rectangular", "Rectangular", "Standard axis-aligned kernel."),
                    ("Elliptical", "Elliptical", "Softer kernel suited to rounded structures."),
                    ("Cross", "Cross", "Thin kernel ideal for preserving thin features."),
                ]
            ),
            value_type=ControlValueType.STRING,
        ),
        "kernel_size": ControlMetadata(
            description="Size of the structuring element in pixels.",
            default=3,
            minimum=1,
            maximum=31,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
        "iterations": ControlMetadata(
            description="Number of erosion passes performed.",
            default=1,
            minimum=1,
            maximum=10,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
    "BorderRemoval": {
        "border_distance": ControlMetadata(
            description="Number of pixels to strip from the image border.",
            default=100,
            minimum=0,
            maximum=999,
            step=1,
            value_type=ControlValueType.INTEGER,
        ),
    },
}


# ----------------------------------------------------------------------
# Lookup helpers for consumers

def get_module_control_metadata(module_identifier: str) -> Mapping[str, ControlMetadata]:
    """Return metadata for ``module_identifier`` or an empty mapping."""

    return MODULE_CONTROL_METADATA.get(module_identifier, {})


def get_control_metadata(module_identifier: str, parameter_name: str) -> ControlMetadata | None:
    """Retrieve metadata for a specific module parameter if registered."""

    return MODULE_CONTROL_METADATA.get(module_identifier, {}).get(parameter_name)


def coerce_parameter(module_identifier: str, parameter_name: str, value: Any | None) -> Any | None:
    """Coerce a value according to the metadata registry."""

    metadata = get_control_metadata(module_identifier, parameter_name)
    if metadata is None:
        return value
    return metadata.coerce(value)


def iter_registered_modules() -> Iterable[str]:
    """Yield the identifiers of all modules with registered metadata."""

    return MODULE_CONTROL_METADATA.keys()


__all__ = [
    "ChoiceOption",
    "ControlMetadata",
    "ControlValueType",
    "MODULE_CONTROL_METADATA",
    "coerce_parameter",
    "get_control_metadata",
    "get_module_control_metadata",
    "iter_registered_modules",
]
