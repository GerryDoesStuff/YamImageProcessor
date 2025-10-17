"""Image I/O helpers for loading and saving arrays with metadata.

This module supports round-tripping image pixel data and associated metadata for
common raster formats. PNG, JPEG, TIFF, and BMP files are loaded via
:mod:`Pillow` so that EXIF data, ICC colour profiles, and other per-image
attributes can be preserved. NumPy ``.npy`` arrays are handled through
:func:`numpy.load` and :func:`numpy.save` to retain dtype and shape information.

Metadata is surfaced to callers through :class:`ImageRecord`, which couples the
pixel :class:`numpy.ndarray` with a free-form ``dict`` of metadata. The loader
captures standard values such as the Pillow image format, mode, size, and the
serialised EXIF/ICC payloads where available. Callers may extend this metadata
with custom keys before saving.

Saving functions mirror the behaviour of :func:`load_image`, attempting to
reuse the previously captured metadata for a faithful round-trip. When saving
to Pillow-supported formats the EXIF and ICC payloads are re-applied if present
and any recognised ``info`` entries (for example DPI) are forwarded. ``.npy``
files store the raw array alongside dtype/shape descriptors for validation when
reloading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageSequence

try:  # pragma: no cover - optional dependency in minimal environments
    import h5py
except Exception:  # pragma: no cover - gracefully handle missing bindings
    h5py = None

from .paths import sanitize_user_path


_SUPPORTED_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_SUPPORTED_VOLUME_SUFFIXES = {".tif", ".tiff", ".h5", ".hdf5", ".hdf"}
_SUPPORTED_ARRAY_SUFFIXES = {".npy", ".npz"}
_DEFAULT_LAZY_PIXEL_THRESHOLD = 64_000_000  # 64 MP ~= 256 MB @ 32-bit RGBA
_LAZY_PIXEL_THRESHOLD = int(
    os.environ.get("YAM_LAZY_PIXEL_THRESHOLD", _DEFAULT_LAZY_PIXEL_THRESHOLD)
)


@dataclass(slots=True)
class ImageRecord:
    """Container coupling image pixel data with associated metadata."""

    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Return the image pixels as a concrete :class:`numpy.ndarray`."""

        return np.asarray(self.data)

    def read_region(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Return a window of pixels defined by ``box``."""

        left, upper, right, lower = _validate_box(box, self.data.shape[:2])
        array = self.to_array()
        if array.ndim == 2:
            return array[upper:lower, left:right]
        return array[upper:lower, left:right, ...]

    def iter_tiles(
        self, tile_size: Tuple[int, int] | None = None
    ) -> Iterator[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """Yield windowed regions of the image.

        Parameters
        ----------
        tile_size:
            Optional ``(width, height)`` describing the stride used to iterate
            over the image. When omitted the full frame is yielded as a single
            tile.
        """

        array = self.to_array()
        for box in _iter_tile_boxes(array.shape[1], array.shape[0], tile_size):
            yield box, self.read_region(box)


@dataclass(slots=True)
class TiledImageRecord:
    """Image record that streams pixel data from a Pillow image handle."""

    image: Image.Image
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cached_array: np.ndarray | None = field(default=None, init=False, repr=False)

    def close(self) -> None:
        """Close the underlying Pillow image handle."""

        self.image.close()

    def to_array(self) -> np.ndarray:
        """Materialise the full image as a :class:`numpy.ndarray`."""

        if self._cached_array is None:
            self._cached_array = _image_to_array(self.image)
        return self._cached_array

    @property
    def data(self) -> np.ndarray:
        """Compatibility accessor mirroring :class:`ImageRecord`."""

        return self.to_array()

    def read_region(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Materialise a window of the image defined by ``box``."""

        left, upper, right, lower = _validate_box(box, self.image.size[::-1])
        region = self.image.crop((left, upper, right, lower))
        return _image_to_array(region)

    def iter_tiles(
        self, tile_size: Tuple[int, int] | None = None
    ) -> Iterator[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """Yield pixel tiles either using Pillow tile metadata or strides."""

        if tile_size is None and getattr(self.image, "tile", None):
            seen: set[Tuple[int, int, int, int]] = set()
            for _decoder, bbox, *_rest in self.image.tile:  # type: ignore[attr-defined]
                if len(bbox) != 4:
                    continue
                if bbox in seen:
                    continue
                seen.add(bbox)
                yield bbox, self.read_region(bbox)
            if seen:
                return

        width, height = self.image.size
        for box in _iter_tile_boxes(width, height, tile_size):
            yield box, self.read_region(box)


@dataclass(slots=True)
class DimensionalImageRecord:
    """Container describing an ``n``-dimensional array with axis metadata."""

    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    dims: Tuple[str, ...] = field(default_factory=tuple)
    coordinates: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.asarray(self.data)

    def axis_size(self, axis: int | str) -> int:
        axis_index = self._resolve_axis(axis)
        return int(self.to_array().shape[axis_index])

    def slice(self, axis: int | str, index: int) -> np.ndarray:
        axis_index = self._resolve_axis(axis)
        slicer = [slice(None)] * self.to_array().ndim
        slicer[axis_index] = int(index)
        return self.to_array()[tuple(slicer)]

    def _resolve_axis(self, axis: int | str) -> int:
        if isinstance(axis, str):
            dims = self._dims()
            if axis not in dims:
                raise KeyError(f"Unknown axis '{axis}'")
            return dims.index(axis)
        return int(axis)

    def _dims(self) -> Tuple[str, ...]:
        if not self.dims:
            self.dims = _infer_dimensions(self.to_array())
        return self.dims


def _normalise_path(path: Path | str) -> Path:
    resolved = sanitize_user_path(path, must_exist=True, allow_directory=False)
    if not resolved.is_file():
        raise IsADirectoryError(resolved)
    return resolved


def load_image(path: Path | str) -> ImageRecord | TiledImageRecord | DimensionalImageRecord:
    """Load an image or ``.npy`` array from ``path``.

    Images that exceed the configured lazy-loading threshold are returned as a
    :class:`TiledImageRecord`, allowing callers to stream tiles without
    materialising the entire array.
    """

    resolved = _normalise_path(path)
    suffix = resolved.suffix.lower()

    if suffix in _SUPPORTED_ARRAY_SUFFIXES:
        array, metadata = _load_numpy(resolved)
        if array.ndim > 3 and not _is_colour_stack(array):
            dims = tuple(metadata.get("dims", _infer_dimensions(array)))
            metadata["dims"] = dims
            return DimensionalImageRecord(data=array, metadata=metadata, dims=dims)
        return ImageRecord(data=array, metadata=metadata)

    if suffix not in _SUPPORTED_RASTER_SUFFIXES:
        if suffix in {".h5", ".hdf5", ".hdf"}:
            return _load_hdf5(resolved)
        raise ValueError(f"Unsupported image format: {resolved.suffix}")

    img = Image.open(resolved)
    metadata = {
        "format": img.format,
        "mode": img.mode,
        "size": img.size,
        "info": dict(img.info),
    }
    exif = img.getexif()
    if exif:
        metadata["exif"] = exif.tobytes()
    icc_profile = img.info.get("icc_profile")
    if icc_profile:
        metadata["icc_profile"] = icc_profile

    if suffix in _SUPPORTED_VOLUME_SUFFIXES and getattr(img, "n_frames", 1) > 1:
        frames = [_image_to_array(frame.copy()) for frame in ImageSequence.Iterator(img)]
        array = np.stack(frames, axis=0)
        dims = _infer_dimensions(array, prefer_volume=True)
        metadata["dims"] = dims
        metadata["frames"] = len(frames)
        img.close()
        return DimensionalImageRecord(data=array, metadata=metadata, dims=dims)

    if _should_stream_image(img):
        return TiledImageRecord(image=img, metadata=metadata)

    try:
        array = _image_to_array(img)
    finally:
        img.close()
    if array.ndim > 3 and not _is_colour_stack(array):
        dims = _infer_dimensions(array)
        metadata["dims"] = dims
        return DimensionalImageRecord(data=array, metadata=metadata, dims=dims)
    return ImageRecord(data=array, metadata=metadata)


def _prepare_save_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    save_args: Dict[str, Any] = {}

    info = metadata.get("info")
    if isinstance(info, Mapping):
        for key, value in info.items():
            if isinstance(key, str) and key not in {"exif", "icc_profile"}:
                save_args[key] = value

    exif = metadata.get("exif")
    if exif is not None:
        save_args["exif"] = exif

    icc_profile = metadata.get("icc_profile")
    if icc_profile is not None:
        save_args["icc_profile"] = icc_profile

    return save_args


def save_image(
    record: ImageRecord | TiledImageRecord | DimensionalImageRecord,
    path: Path | str,
    format: Optional[str] = None,
) -> None:
    """Persist ``record`` to ``path`` using ``format`` if supplied."""

    destination = sanitize_user_path(path, must_exist=False, allow_directory=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fmt = (format or destination.suffix.lstrip(".")).upper()
    if fmt in {"NPY", "NPZ"}:
        _save_numpy(record, destination, fmt)
        return

    if fmt in {"H5", "HDF5"}:
        _save_hdf5(record, destination)
        return

    if fmt == "TIFF" and isinstance(record, DimensionalImageRecord):
        _save_tiff_stack(record, destination)
        return

    # For raster formats defer to Pillow for encoding.
    if destination.suffix.lower() not in _SUPPORTED_RASTER_SUFFIXES and fmt.lower() not in {
        s.lstrip(".") for s in _SUPPORTED_RASTER_SUFFIXES
    }:
        raise ValueError(f"Unsupported image format for saving: {fmt}")

    array = np.asarray(record.to_array())
    image = Image.fromarray(array)
    if isinstance(record.metadata, Mapping):
        target_mode = record.metadata.get("mode")
        if isinstance(target_mode, str) and target_mode != image.mode:
            try:
                image = image.convert(target_mode)
            except ValueError:
                # Ignore invalid conversions and keep the Pillow-derived mode.
                pass
        save_kwargs = _prepare_save_metadata(record.metadata)
    else:
        save_kwargs = {}

    image.save(destination, format=format, **save_kwargs)

    if isinstance(record.metadata, MutableMapping):
        record.metadata.setdefault("format", image.format or fmt)
        record.metadata.setdefault("mode", image.mode)
        record.metadata.setdefault("size", image.size)


def _load_numpy(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "arr_0" not in archive:
                raise ValueError(".npz archives must contain an 'arr_0' dataset")
            array = archive["arr_0"]
            metadata_raw = archive.get("metadata")
            if metadata_raw is None:
                metadata = {}
            else:
                try:
                    metadata = json.loads(metadata_raw.item())  # type: ignore[arg-type]
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError("Failed to decode metadata from .npz archive") from exc
    else:
        array = np.load(path, allow_pickle=False)
        metadata = {}
    metadata.setdefault("format", "NPY")
    metadata.setdefault("dtype", str(array.dtype))
    metadata.setdefault("shape", array.shape)
    return array, metadata


def _save_numpy(
    record: ImageRecord | TiledImageRecord | DimensionalImageRecord, path: Path, fmt: str
) -> None:
    array = np.asarray(record.to_array())
    metadata = dict(record.metadata) if isinstance(record.metadata, MutableMapping) else {}
    metadata.setdefault("format", fmt)
    metadata.setdefault("dtype", str(array.dtype))
    metadata.setdefault("shape", array.shape)
    if isinstance(record, DimensionalImageRecord) and record.dims:
        metadata.setdefault("dims", tuple(record.dims))
    if fmt == "NPZ":
        payload: Dict[str, np.ndarray] = {"arr_0": array}
        payload["metadata"] = np.array(json.dumps(metadata))
        np.savez(path, **payload)
    else:
        np.save(path, array, allow_pickle=False)


def _load_hdf5(path: Path) -> DimensionalImageRecord:
    if h5py is None:
        raise RuntimeError("h5py is required to load HDF5 files but is not installed")
    with h5py.File(path, "r") as handle:  # type: ignore[call-arg]
        dataset = _select_first_dataset(handle)
        array = dataset[()]
        metadata = {
            "format": "HDF5",
            "dtype": str(array.dtype),
            "shape": array.shape,
            "path": dataset.name,
        }
        dims_attr = dataset.attrs.get("dims")
        if isinstance(dims_attr, (list, tuple)):
            dims = tuple(str(dim) for dim in dims_attr)
        else:
            dims = _infer_dimensions(array)
        metadata["dims"] = dims
        coordinates: Dict[str, np.ndarray] = {}
        for key, value in dataset.attrs.items():
            if key.startswith("coord_"):
                axis = key.split("coord_", 1)[1]
                coordinates[axis] = np.array(value)
    return DimensionalImageRecord(data=array, metadata=metadata, dims=dims, coordinates=coordinates)


def _save_hdf5(
    record: ImageRecord | TiledImageRecord | DimensionalImageRecord, path: Path
) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to save HDF5 files but is not installed")
    array = np.asarray(record.to_array())
    dims = _infer_record_dims(record, array)
    with h5py.File(path, "w") as handle:  # type: ignore[call-arg]
        dataset = handle.create_dataset("data", data=array)
        dataset.attrs["dims"] = list(dims)
        if isinstance(record, DimensionalImageRecord):
            for axis, values in record.coordinates.items():
                dataset.attrs[f"coord_{axis}"] = np.asarray(values)


def _save_tiff_stack(record: DimensionalImageRecord, path: Path) -> None:
    array = np.asarray(record.to_array())
    if array.ndim < 3:
        raise ValueError("TIFF stacks require arrays with three or more dimensions")
    frames = []
    for index in range(array.shape[0]):
        frame = array[index]
        if frame.ndim == 2:
            frames.append(Image.fromarray(frame))
        elif frame.ndim == 3 and frame.shape[2] in (1, 3, 4):
            frames.append(Image.fromarray(frame))
        else:
            raise ValueError("Cannot serialise multi-channel slice with unsupported shape")
    first, *rest = frames
    first.save(path, save_all=True, append_images=rest)


def _select_first_dataset(handle: "h5py.File"):
    queue: list[Any] = [handle]
    while queue:
        current = queue.pop(0)
        if isinstance(current, h5py.Dataset):
            return current
        for value in current.values():
            if isinstance(value, (h5py.Dataset, h5py.Group)):
                queue.append(value)
    raise ValueError("No datasets found in HDF5 file")


def _infer_record_dims(
    record: ImageRecord | TiledImageRecord | DimensionalImageRecord, array: np.ndarray
) -> Tuple[str, ...]:
    if isinstance(record, DimensionalImageRecord) and record.dims:
        return tuple(record.dims)
    metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
    dims = metadata.get("dims")
    if isinstance(dims, (list, tuple)):
        return tuple(str(dim) for dim in dims)
    return _infer_dimensions(array)


def _infer_dimensions(array: np.ndarray, *, prefer_volume: bool = False) -> Tuple[str, ...]:
    if array.ndim == 2:
        return ("y", "x")
    if array.ndim == 3:
        if _is_colour_stack(array) and not prefer_volume:
            return ("y", "x", "channel")
        return ("z", "y", "x")
    dims = [f"dim{i}" for i in range(array.ndim - 2)] + ["y", "x"]
    return tuple(dims)


def _is_colour_stack(array: np.ndarray) -> bool:
    return array.ndim == 3 and array.shape[2] in (3, 4)


def _image_to_array(image: Image.Image) -> np.ndarray:
    """Convert ``image`` into a :class:`numpy.ndarray`."""

    return np.array(image)


def _should_stream_image(image: Image.Image) -> bool:
    """Determine if ``image`` should be streamed instead of fully materialised."""

    width, height = image.size
    threshold = max(0, _LAZY_PIXEL_THRESHOLD)
    if threshold and width * height >= threshold:
        return True

    # Some formats advertise their tiling characteristics, which is a good
    # signal that random access is beneficial even for smaller frames.
    if getattr(image, "tile", None):  # type: ignore[attr-defined]
        try:
            first_tile = image.tile[0]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive against Pillow internals
            first_tile = None
        if first_tile and len(first_tile) >= 2:
            _decoder, bbox = first_tile[:2]
            if isinstance(bbox, tuple) and len(bbox) == 4:
                left, upper, right, lower = bbox
                if (right - left) * (lower - upper) < width * height:
                    return True
    return False


def _validate_box(box: Tuple[int, int, int, int], shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Normalise and validate a Pillow-style bounding box."""

    if len(box) != 4:
        raise ValueError("Expected 4-tuple bounding box")
    left, upper, right, lower = (int(value) for value in box)
    height, width = shape_hw
    if not (0 <= left < right <= width and 0 <= upper < lower <= height):
        raise ValueError(
            f"Bounding box {box!r} outside the image domain {width}x{height}."
        )
    return left, upper, right, lower


def _iter_tile_boxes(
    width: int, height: int, tile_size: Tuple[int, int] | None
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield bounding boxes that cover an image using strides."""

    if tile_size is None:
        step_x, step_y = width, height
    else:
        step_x, step_y = tile_size
        if step_x <= 0 or step_y <= 0:
            raise ValueError("Tile dimensions must be positive integers")
    for top in range(0, height, step_y):
        bottom = min(top + step_y, height)
        for left in range(0, width, step_x):
            right = min(left + step_x, width)
            yield (left, top, right, bottom)
