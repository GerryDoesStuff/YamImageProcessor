"""Unified application settings management built on top of ``QSettings``."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - Qt availability differs per environment
    from PyQt5.QtCore import QSettings  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PySide6.QtCore import QSettings  # type: ignore
    except Exception:  # pragma: no cover
        QSettings = None  # type: ignore


DEFAULT_SETTINGS: Dict[str, Any] = {
    # Diagnostics -----------------------------------------------------------------
    "diagnostics/enabled": False,
    # Preprocessing ---------------------------------------------------------------
    "preprocess/order": "",
    "preprocess/grayscale": False,
    "preprocess/brightness_contrast/enabled": False,
    "preprocess/brightness_contrast/alpha": 1.0,
    "preprocess/brightness_contrast/beta": 0,
    "preprocess/gamma/enabled": False,
    "preprocess/gamma/value": 1.0,
    "preprocess/normalize/enabled": False,
    "preprocess/normalize/alpha": 0,
    "preprocess/normalize/beta": 255,
    "preprocess/noise_reduction/enabled": False,
    "preprocess/noise_reduction/method": "Gaussian",
    "preprocess/noise_reduction/ksize": 5,
    "preprocess/sharpen/enabled": False,
    "preprocess/sharpen/strength": 1.0,
    "preprocess/select_channel/enabled": False,
    "preprocess/select_channel/value": "All",
    "preprocess/crop/enabled": False,
    "preprocess/crop/x_offset": 0,
    "preprocess/crop/y_offset": 0,
    "preprocess/crop/width": 100,
    "preprocess/crop/height": 100,
    # Segmentation ----------------------------------------------------------------
    "segmentation/order": "",
    "segmentation/Global/enabled": False,
    "segmentation/Global/threshold": 127,
    "segmentation/Otsu/enabled": False,
    "segmentation/Adaptive/enabled": False,
    "segmentation/Adaptive/block_size": 11,
    "segmentation/Adaptive/C": 2,
    "segmentation/Edge/enabled": False,
    "segmentation/Edge/low_threshold": 50,
    "segmentation/Edge/high_threshold": 150,
    "segmentation/Edge/aperture_size": 3,
    "segmentation/Watershed/enabled": False,
    "segmentation/Watershed/kernel_size": 3,
    "segmentation/Watershed/opening_iterations": 2,
    "segmentation/Watershed/dilation_iterations": 3,
    "segmentation/Watershed/distance_threshold_factor": 0.7,
    "segmentation/Sobel/enabled": False,
    "segmentation/Sobel/ksize": 3,
    "segmentation/Prewitt/enabled": False,
    "segmentation/Laplacian/enabled": False,
    "segmentation/Laplacian/ksize": 3,
    "segmentation/Region Growing/enabled": False,
    "segmentation/Region Growing/seed_x": 50,
    "segmentation/Region Growing/seed_y": 50,
    "segmentation/Region Growing/tolerance": 10,
    "segmentation/Region Splitting/Merging/enabled": False,
    "segmentation/Region Splitting/Merging/min_size": 16,
    "segmentation/Region Splitting/Merging/std_thresh": 10.0,
    "segmentation/K-Means/enabled": False,
    "segmentation/K-Means/K": 2,
    "segmentation/K-Means/seed": 42,
    "segmentation/Fuzzy C-Means/enabled": False,
    "segmentation/Fuzzy C-Means/K": 2,
    "segmentation/Fuzzy C-Means/seed": 42,
    "segmentation/Mean Shift/enabled": False,
    "segmentation/Mean Shift/spatial_radius": 20,
    "segmentation/Mean Shift/color_radius": 30,
    "segmentation/GMM/enabled": False,
    "segmentation/GMM/components": 2,
    "segmentation/GMM/seed": 42,
    "segmentation/Graph Cuts/enabled": False,
    "segmentation/Active Contour/enabled": False,
    "segmentation/Active Contour/iterations": 250,
    "segmentation/Active Contour/alpha": 0.015,
    "segmentation/Active Contour/beta": 10.0,
    "segmentation/Active Contour/gamma": 0.001,
    "segmentation/Opening/enabled": False,
    "segmentation/Opening/kernel_shape": "Rectangular",
    "segmentation/Opening/kernel_size": 3,
    "segmentation/Opening/iterations": 1,
    "segmentation/Closing/enabled": False,
    "segmentation/Closing/kernel_shape": "Rectangular",
    "segmentation/Closing/kernel_size": 3,
    "segmentation/Closing/iterations": 1,
    "segmentation/Dilation/enabled": False,
    "segmentation/Dilation/kernel_shape": "Rectangular",
    "segmentation/Dilation/kernel_size": 3,
    "segmentation/Dilation/iterations": 1,
    "segmentation/Erosion/enabled": False,
    "segmentation/Erosion/kernel_shape": "Rectangular",
    "segmentation/Erosion/kernel_size": 3,
    "segmentation/Erosion/iterations": 1,
    "segmentation/Border Removal/enabled": False,
    "segmentation/Border Removal/border_distance": 25,
    # Extraction -------------------------------------------------------------------
    "extraction/order": "",
    "extraction/Region Properties/enabled": False,
    "extraction/Hu Moments/enabled": False,
    "extraction/LBP/enabled": False,
    "extraction/LBP/P": 8,
    "extraction/LBP/R": 1.0,
    "extraction/Haralick/enabled": False,
    "extraction/Haralick/distance": 1,
    "extraction/Haralick/angle": 0.0,
    "extraction/Gabor/enabled": False,
    "extraction/Gabor/ksize": 21,
    "extraction/Gabor/sigma": 5.0,
    "extraction/Gabor/theta": 0.0,
    "extraction/Gabor/lambd": 10.0,
    "extraction/Gabor/gamma": 0.5,
    "extraction/Gabor/psi": 0.0,
    "extraction/Fourier/enabled": False,
    "extraction/Fourier/num_coeff": 10,
    "extraction/HOG/enabled": False,
    "extraction/HOG/orientations": 9,
    "extraction/HOG/ppc": 8,
    "extraction/HOG/cpb": 3,
    "extraction/Histogram/enabled": False,
    "extraction/Fractal/enabled": False,
    "extraction/Fractal/min_box_size": 2,
    "extraction/Approximate Shape/enabled": False,
    "extraction/Approximate Shape/error_threshold": 1.0,
}


class _FallbackSettings:
    """In-memory substitute mirroring the ``QSettings`` API."""

    def __init__(self, organization: str, application: str) -> None:
        self._organization = organization
        self._application = application
        self._store: Dict[str, Any] = {}

    # QSettings compatibility -----------------------------------------------------
    def setValue(self, key: str, value: Any) -> None:  # noqa: N802 - Qt naming
        self._store[key] = value

    def value(self, key: str, default: Any | None = None) -> Any:  # noqa: N802 - Qt naming
        return self._store.get(key, default)

    def remove(self, key: str) -> None:  # noqa: N802 - Qt naming
        self._store.pop(key, None)

    def contains(self, key: str) -> bool:  # noqa: N802 - Qt naming
        return key in self._store

    def allKeys(self) -> Iterable[str]:  # noqa: N802 - Qt naming
        return list(self._store.keys())

    def clear(self) -> None:  # noqa: N802 - Qt naming
        self._store.clear()

    def sync(self) -> None:  # noqa: N802 - Qt naming, pragma: no cover - noop for fallback
        return None


class SettingsManager:
    """High level wrapper for ``QSettings`` that supports JSON import/export."""

    def __init__(
        self,
        organization: str,
        application: str,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        seed_defaults: bool = True,
    ) -> None:
        backend = QSettings if QSettings is not None else _FallbackSettings
        self._settings = backend(organization, application)
        self.organization = organization
        self.application = application
        self._defaults: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        if defaults:
            self._defaults.update(defaults)
        if seed_defaults:
            self.seed_defaults()

    # ------------------------------------------------------------------
    # Core CRUD helpers
    def get(self, key: str, default: Any | None = None) -> Any:
        return self._settings.value(key, default)

    def set(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)
        self._settings.sync()

    def remove(self, key: str) -> None:
        self._settings.remove(key)
        self._settings.sync()

    def contains(self, key: str) -> bool:
        return bool(getattr(self._settings, "contains", lambda k: False)(key))

    def clear(self, prefix: str | None = None) -> None:
        if prefix is None:
            self._settings.clear()
        else:
            for key in list(self._all_keys(prefix=prefix)):
                self._settings.remove(key)
        self._settings.sync()

    def sync(self) -> None:
        self._settings.sync()

    # ------------------------------------------------------------------
    # Default management
    def seed_defaults(self) -> None:
        for key, value in self._defaults.items():
            if not self.contains(key):
                self._settings.setValue(key, value)
        self._settings.sync()

    # ------------------------------------------------------------------
    # Snapshot helpers
    def snapshot(
        self,
        *,
        prefix: str | None = None,
        strip_prefix: bool = False,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key in self._all_keys(prefix=prefix):
            store_key = key
            if prefix and strip_prefix:
                store_key = key[len(prefix) :]
            data[store_key] = self._settings.value(key)
        return data

    def to_dict(
        self,
        *,
        prefix: str | None = None,
        strip_prefix: bool = False,
    ) -> Dict[str, Any]:
        return self.snapshot(prefix=prefix, strip_prefix=strip_prefix)

    def apply_snapshot(
        self,
        values: Mapping[str, Any],
        *,
        prefix: str | None = None,
        clear: bool = False,
    ) -> None:
        if clear:
            self.clear(prefix=prefix)
        for key, value in values.items():
            full_key = key
            if prefix and not key.startswith(prefix):
                full_key = f"{prefix}{key}"
            self._settings.setValue(full_key, value)
        self._settings.sync()

    def from_dict(
        self,
        values: Mapping[str, Any],
        *,
        prefix: str | None = None,
        clear: bool = False,
    ) -> None:
        self.apply_snapshot(values, prefix=prefix, clear=clear)

    # ------------------------------------------------------------------
    # JSON helpers
    def to_json(
        self,
        *,
        prefix: str | None = None,
        strip_prefix: bool = False,
        indent: int = 2,
    ) -> str:
        snapshot = self.snapshot(prefix=prefix, strip_prefix=strip_prefix)
        return json.dumps(snapshot, indent=indent, sort_keys=True)

    def from_json(
        self,
        payload: str | Mapping[str, Any],
        *,
        prefix: str | None = None,
        clear: bool = False,
    ) -> None:
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = dict(payload)
        if not isinstance(data, MutableMapping):
            raise ValueError("Settings JSON must describe an object")
        self.apply_snapshot(data, prefix=prefix, clear=clear)

    def export_json(
        self,
        path: Path,
        *,
        prefix: str | None = None,
        strip_prefix: bool = False,
        indent: int = 2,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json(prefix=prefix, strip_prefix=strip_prefix, indent=indent)
        path.write_text(payload, encoding="utf-8")

    def import_json(
        self,
        path: Path,
        *,
        prefix: str | None = None,
        clear: bool = False,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = path.read_text(encoding="utf-8")
        self.from_json(payload, prefix=prefix, clear=clear)

    # ------------------------------------------------------------------
    @property
    def backend(self) -> Any:
        return self._settings

    def _all_keys(self, prefix: str | None = None) -> Iterable[str]:
        keys: Iterable[str]
        if hasattr(self._settings, "allKeys"):
            keys = self._settings.allKeys()
        else:  # pragma: no cover - minimal compatibility shim
            keys = []
        if prefix is None:
            return list(keys)
        return [key for key in keys if key.startswith(prefix)]


__all__ = ["SettingsManager", "DEFAULT_SETTINGS"]

