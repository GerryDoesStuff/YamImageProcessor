"""Caching utilities for pipeline step results and reproducibility metadata."""
from __future__ import annotations

from dataclasses import dataclass
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from core.settings import SettingsManager
from core.thread_controller import OperationCancelled
from processing.pipeline_manager import PipelineStep
from processing.tiled_records import TiledPipelineImage, TileSize
from core.tiled_image import TileBox

try:  # pragma: no cover - defensive fallback when numpy stubs lack ndarray
    NDArray = np.ndarray
except AttributeError:  # pragma: no cover - executed in minimal test environments
    NDArray = type(None)


def _normalise_value(value: Any) -> Any:
    """Return a JSON-serialisable representation of ``value``."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_normalise_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalise_value(value[key]) for key in sorted(value)}
    return repr(value)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    """Generate a stable SHA256 hash for ``payload``."""

    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(serialised).hexdigest()
    return digest


@dataclass(frozen=True)
class StepRecord:
    """Metadata describing a single cached pipeline step."""

    name: str
    enabled: bool
    params: Dict[str, Any]
    signature: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "params": {key: _normalise_value(value) for key, value in self.params.items()},
            "signature": self.signature,
            "index": self.index,
        }


@dataclass
class PipelineCacheResult:
    """Container returned when a cached pipeline evaluation completes."""

    source_id: str
    final_signature: str
    image: NDArray
    steps: List[StepRecord]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class PipelineCacheTileUpdate:
    """Incremental update emitted while streaming tiled pipeline results."""

    source_id: str
    final_signature: str
    step_signature: str
    step_index: int
    total_steps: int
    box: TileBox
    tile: NDArray
    shape: Tuple[int, ...]
    dtype: np.dtype
    tile_size: Optional[TileSize]
    from_cache: bool = False


CacheValue = Union[NDArray, "TileCacheEntry"]


@dataclass
class TileCacheEntry:
    """Container storing per-tile pipeline outputs."""

    shape: Tuple[int, ...]
    dtype: np.dtype
    tiles: List[Tuple[TileBox, NDArray]]
    tile_size: Optional[TileSize] = None

    def iter_tiles(self) -> Iterator[Tuple[TileBox, NDArray]]:
        for box, tile in self.tiles:
            yield box, np.array(tile, copy=True)

    def assemble(self) -> NDArray:
        result = np.zeros(self.shape, dtype=self.dtype)
        for box, tile in self.tiles:
            left, top, right, bottom = box
            if result.ndim == 2:
                result[top:bottom, left:right] = tile
            else:
                result[top:bottom, left:right, ...] = tile
        return result

    @classmethod
    def from_tiles(
        cls,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        tiles: List[Tuple[TileBox, NDArray]],
        *,
        tile_size: Optional[TileSize] = None,
    ) -> "TileCacheEntry":
        copied = [(box, np.array(tile, copy=True)) for box, tile in tiles]
        return cls(shape=tuple(shape), dtype=np.dtype(dtype), tiles=copied, tile_size=tile_size)

    @classmethod
    def from_array(cls, array: NDArray) -> "TileCacheEntry":
        if array.ndim < 2:
            raise ValueError("TileCacheEntry requires arrays with at least two dimensions")
        height, width = array.shape[0], array.shape[1]
        box: TileBox = (0, 0, int(width), int(height))
        return cls(
            shape=tuple(array.shape),
            dtype=array.dtype,
            tiles=[(box, np.array(array, copy=True))],
            tile_size=(int(width), int(height)),
        )

class PipelineCache:
    """Store step outputs keyed by source image and pipeline configuration."""

    SETTINGS_KEY = "pipeline_cache/state"

    _DEFAULT_CACHE_DIRECTORY: Optional[Path] = None

    def __init__(
        self,
        settings: Optional[SettingsManager] = None,
        *,
        cache_directory: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self._logger = logging.getLogger(f"{__name__}.PipelineCache")
        self._settings = settings
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._cache: Dict[str, Dict[str, CacheValue]] = {}
        self._lock = threading.Lock()
        self._cache_directory: Optional[Path] = None
        self.set_cache_directory(
            cache_directory if cache_directory is not None else self._DEFAULT_CACHE_DIRECTORY
        )
        self._load_metadata()

    @classmethod
    def set_default_cache_directory(cls, path: Optional[os.PathLike[str] | str]) -> None:
        cls._DEFAULT_CACHE_DIRECTORY = None if path is None else Path(path)
        if cls._DEFAULT_CACHE_DIRECTORY is not None:
            cls._DEFAULT_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    @property
    def cache_directory(self) -> Optional[Path]:
        return self._cache_directory

    def set_cache_directory(self, path: Optional[os.PathLike[str] | str]) -> None:
        if path is None:
            if self._cache_directory is not None:
                self._logger.debug(
                    "Disabling on-disk pipeline cache",
                    extra={"directory": str(self._cache_directory)},
                )
            self._cache_directory = None
            return

        directory = Path(path)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning(
                "Failed to initialise pipeline cache directory",
                extra={"directory": str(directory), "error": str(exc)},
            )
            self._cache_directory = None
            return

        self._cache_directory = directory
        self._logger.debug(
            "Configured pipeline cache directory",
            extra={"directory": str(directory)},
        )

    # ------------------------------------------------------------------
    # Public helpers
    def register_source(self, image: NDArray, *, hint: Optional[str] = None) -> str:
        """Register ``image`` and return a deterministic source identifier."""

        array = np.ascontiguousarray(image)
        h = hashlib.sha256()
        h.update(str(array.shape).encode("utf-8"))
        h.update(str(array.dtype).encode("utf-8"))
        h.update(array.tobytes())
        source_id = h.hexdigest()

        with self._lock:
            cache = self._cache.setdefault(source_id, {})
            stored = np.array(array, copy=True)
            cache[source_id] = stored
            metadata = {
                "version": 1,
                "source_id": source_id,
                "final_signature": source_id,
                "steps": [],
            }
            if hint:
                metadata["hint"] = str(hint)
            self._metadata.setdefault(source_id, {})[source_id] = metadata
            self._persist_metadata_locked()
            self._write_disk_cache(source_id, source_id, stored)

        return source_id

    def discard_cache(self, source_id: str) -> None:
        """Remove in-memory cache entries for ``source_id`` (metadata retained)."""

        with self._lock:
            self._cache.pop(source_id, None)
            self._remove_disk_cache(source_id)

    def predict(self, source_id: str, steps: Sequence[PipelineStep]) -> Tuple[str, List[StepRecord]]:
        """Return the expected final signature and per-step metadata."""

        signature = source_id
        records: List[StepRecord] = []
        for index, step in enumerate(steps):
            payload = {
                "previous": signature,
                "name": step.name,
                "enabled": bool(step.enabled),
                "params": _normalise_value(step.params),
            }
            signature = _hash_payload(payload)
            records.append(
                StepRecord(
                    name=step.name,
                    enabled=bool(step.enabled),
                    params=dict(step.params),
                    signature=signature,
                    index=index,
                )
            )
        return signature, records

    def compute(
        self,
        source_id: str,
        image: NDArray | TiledPipelineImage,
        steps: Sequence[PipelineStep],
        *,
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[Callable[[int], None]] = None,
        incremental: Optional[Callable[[PipelineCacheTileUpdate], None]] = None,
    ) -> PipelineCacheResult:
        """Evaluate ``steps`` against ``image`` storing intermediate results."""

        final_signature, records = self.predict(source_id, steps)

        if isinstance(image, TiledPipelineImage):
            return self._compute_tiled(
                source_id,
                image,
                steps,
                final_signature,
                records,
                cancel_event=cancel_event,
                progress=progress,
                incremental=incremental,
            )

        return self._compute_dense(
            source_id,
            image,
            steps,
            final_signature,
            records,
            cancel_event=cancel_event,
            progress=progress,
            incremental=incremental,
        )

    def _compute_dense(
        self,
        source_id: str,
        image: NDArray,
        steps: Sequence[PipelineStep],
        final_signature: str,
        records: List[StepRecord],
        *,
        cancel_event: Optional[threading.Event],
        progress: Optional[Callable[[int], None]],
        incremental: Optional[Callable[[PipelineCacheTileUpdate], None]],
    ) -> PipelineCacheResult:
        with self._lock:
            cache = self._cache.setdefault(source_id, {})

        total_steps = max(1, len(steps))
        result = np.array(image, copy=True)

        for index, (step, record) in enumerate(zip(steps, records)):
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled()
            with self._lock:
                cached = cache.get(record.signature)
            if cached is not None:
                result = np.array(self._coerce_cache_to_array(cached), copy=True)
            else:
                if step.enabled:
                    result = step.apply(result)
                else:
                    result = np.array(result, copy=True)
                with self._lock:
                    stored = np.array(result, copy=True)
                    cache[record.signature] = stored
                    self._write_disk_cache(source_id, record.signature, stored)
            if progress is not None:
                progress(int(((index + 1) / total_steps) * 100))

        if not records:
            with self._lock:
                stored = np.array(result, copy=True)
                cache[final_signature] = stored
                self._write_disk_cache(source_id, final_signature, stored)

        metadata = {
            "version": 1,
            "source_id": source_id,
            "final_signature": final_signature,
            "steps": [record.to_dict() for record in records],
        }

        with self._lock:
            self._metadata.setdefault(source_id, {})[final_signature] = metadata
            self._persist_metadata_locked()

        return PipelineCacheResult(
            source_id=source_id,
            final_signature=final_signature,
            image=np.array(result, copy=True),
            steps=records,
            metadata=json.loads(json.dumps(metadata)),
        )

    def _compute_tiled(
        self,
        source_id: str,
        image: TiledPipelineImage,
        steps: Sequence[PipelineStep],
        final_signature: str,
        records: List[StepRecord],
        *,
        cancel_event: Optional[threading.Event],
        progress: Optional[Callable[[int], None]],
        incremental: Optional[Callable[[PipelineCacheTileUpdate], None]],
    ) -> PipelineCacheResult:
        with self._lock:
            cache = self._cache.setdefault(source_id, {})

        total_steps = max(1, len(steps))
        shape = image.infer_shape()
        dtype_hint = image.dtype or np.float32
        tile_size = image.tile_size
        current_entry: Optional[TileCacheEntry] = None

        def _emit_tile_update(
            *,
            box: TileBox,
            tile: NDArray,
            step_signature: str,
            step_index: int,
            from_cache: bool = False,
        ) -> None:
            if incremental is None:
                return
            payload = PipelineCacheTileUpdate(
                source_id=source_id,
                final_signature=final_signature,
                step_signature=step_signature,
                step_index=step_index,
                total_steps=total_steps,
                box=tuple(int(value) for value in box),
                tile=np.array(tile, copy=True),
                shape=tuple(int(dim) for dim in shape),
                dtype=np.dtype(tile.dtype),
                tile_size=tile_size,
                from_cache=from_cache,
            )
            incremental(payload)

        for index, (step, record) in enumerate(zip(steps, records)):
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled()

            cached = self._get_cached_value(cache, source_id, record.signature)
            if cached is not None:
                if isinstance(cached, NDArray):
                    current_entry = TileCacheEntry.from_array(cached)
                else:
                    current_entry = cached
                if (
                    incremental is not None
                    and current_entry is not None
                    and index + 1 == total_steps
                ):
                    for box, tile in current_entry.iter_tiles():
                        _emit_tile_update(
                            box=box,
                            tile=tile,
                            step_signature=record.signature,
                            step_index=index + 1,
                            from_cache=True,
                        )
            else:
                iterator: Iterator[Tuple[TileBox, NDArray]]
                if current_entry is None:
                    iterator = image.iter_tiles(tile_size)
                else:
                    iterator = current_entry.iter_tiles()

                tiles: List[Tuple[TileBox, NDArray]] = []
                tile_dtype: Optional[np.dtype] = None
                for box, tile in iterator:
                    operand = np.array(tile, copy=True)
                    result = step.apply(operand) if step.enabled else operand
                    if isinstance(result, TiledPipelineImage):
                        result = result.to_array()
                    tile_array = np.array(result, copy=False)
                    if tile_dtype is None:
                        tile_dtype = tile_array.dtype
                    tiles.append((box, np.array(tile_array, copy=True)))
                    if index + 1 == total_steps:
                        _emit_tile_update(
                            box=box,
                            tile=tile_array,
                            step_signature=record.signature,
                            step_index=index + 1,
                        )

                if not tiles:
                    tile_dtype = np.dtype(dtype_hint)
                elif tile_dtype is None:
                    tile_dtype = np.dtype(dtype_hint)

                current_entry = TileCacheEntry.from_tiles(
                    shape,
                    tile_dtype,
                    tiles,
                    tile_size=tile_size,
                )

                with self._lock:
                    cache[record.signature] = current_entry
                    self._write_disk_cache(source_id, record.signature, current_entry)

            if progress is not None:
                progress(int(((index + 1) / total_steps) * 100))

        if current_entry is None:
            tiles: List[Tuple[TileBox, NDArray]] = []
            tile_dtype: Optional[np.dtype] = None
            for box, tile in image.iter_tiles(tile_size):
                array_tile = np.array(tile, copy=True)
                tiles.append((box, array_tile))
                if tile_dtype is None:
                    tile_dtype = array_tile.dtype
                _emit_tile_update(
                    box=box,
                    tile=array_tile,
                    step_signature=final_signature,
                    step_index=total_steps,
                )
            if tile_dtype is None:
                tile_dtype = np.dtype(dtype_hint)
            current_entry = TileCacheEntry.from_tiles(
                shape,
                tile_dtype,
                tiles,
                tile_size=tile_size,
            )
            with self._lock:
                cache[final_signature] = current_entry
                self._write_disk_cache(source_id, final_signature, current_entry)

        metadata = {
            "version": 1,
            "source_id": source_id,
            "final_signature": final_signature,
            "steps": [record.to_dict() for record in records],
        }

        with self._lock:
            self._metadata.setdefault(source_id, {})[final_signature] = metadata
            self._persist_metadata_locked()

        image_array = current_entry.assemble()
        return PipelineCacheResult(
            source_id=source_id,
            final_signature=final_signature,
            image=np.array(image_array, copy=True),
            steps=records,
            metadata=json.loads(json.dumps(metadata)),
        )

    def _coerce_cache_to_array(self, value: CacheValue) -> NDArray:
        if isinstance(value, TileCacheEntry):
            return value.assemble()
        return value

    def _get_cached_value(
        self,
        cache: Dict[str, CacheValue],
        source_id: str,
        signature: str,
    ) -> Optional[CacheValue]:
        cached = cache.get(signature)
        if cached is not None:
            return cached
        disk_cached = self._load_disk_cache(source_id, signature)
        if disk_cached is None:
            return None
        with self._lock:
            cache[signature] = disk_cached
        return disk_cached

    def get_cached_image(self, source_id: str, signature: str) -> Optional[NDArray]:
        """Return a cached copy of the array for ``signature`` if available."""

        with self._lock:
            cache = self._cache.get(source_id)
            cached = cache.get(signature) if cache else None
            if cached is not None:
                array = self._coerce_cache_to_array(cached)
                return np.array(array, copy=True)
        disk_cached = self._load_disk_cache(source_id, signature)
        if disk_cached is None:
            return None
        with self._lock:
            cache = self._cache.setdefault(source_id, {})
            cache[signature] = disk_cached
        array = self._coerce_cache_to_array(disk_cached)
        return np.array(array, copy=True)

    def metadata_for(self, source_id: str, signature: Optional[str]) -> Dict[str, Any]:
        """Return stored metadata for ``signature`` if available."""

        if signature is None:
            return {}
        with self._lock:
            data = self._metadata.get(source_id, {}).get(signature)
            if data is None:
                return {}
            return json.loads(json.dumps(data))

    # ------------------------------------------------------------------
    # Persistence helpers
    def _load_metadata(self) -> None:
        if self._settings is None:
            self._metadata = {}
        else:
            payload = self._settings.get(self.SETTINGS_KEY, "{}")
            data: Dict[str, Dict[str, Dict[str, Any]]] = {}
            if isinstance(payload, str):
                try:
                    loaded = json.loads(payload)
                except json.JSONDecodeError:
                    loaded = {}
            elif isinstance(payload, MutableMapping):
                loaded = dict(payload)
            else:
                loaded = {}
            for source_id, entries in loaded.items():
                if isinstance(entries, MutableMapping):
                    data[source_id] = {
                        key: dict(value)
                        for key, value in entries.items()
                        if isinstance(value, MutableMapping)
                    }
            self._metadata = data

        disk_metadata = self._load_metadata_from_disk()
        if disk_metadata:
            if not self._metadata:
                self._metadata = disk_metadata
            else:
                for source_id, entries in disk_metadata.items():
                    existing = self._metadata.setdefault(source_id, {})
                    existing.update(entries)

    def _persist_metadata_locked(self) -> None:
        if self._settings is not None:
            serialisable = json.dumps(self._metadata, sort_keys=True)
            self._settings.set(self.SETTINGS_KEY, serialisable)
        self._write_metadata_snapshot_locked()

    def _metadata_snapshot_path(self) -> Optional[Path]:
        if self._cache_directory is None:
            return None
        return self._cache_directory / "metadata.json"

    def _write_metadata_snapshot_locked(self) -> None:
        path = self._metadata_snapshot_path()
        if path is None:
            return
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(self._metadata, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug(
                "Failed to persist pipeline cache metadata snapshot",
                extra={"path": str(path), "error": str(exc)},
            )
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

    def _load_metadata_from_disk(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        path = self._metadata_snapshot_path()
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug(
                "Failed to read pipeline cache metadata snapshot",
                extra={"path": str(path), "error": str(exc)},
            )
            return {}
        data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if isinstance(payload, MutableMapping):
            for source_id, entries in payload.items():
                if isinstance(entries, MutableMapping):
                    data[source_id] = {
                        key: dict(value)
                        for key, value in entries.items()
                        if isinstance(value, MutableMapping)
                    }
        return data

    def _write_disk_cache(self, source_id: str, signature: str, value: CacheValue) -> None:
        directory = self._cache_directory
        if directory is None:
            return
        if isinstance(value, TileCacheEntry):
            path = directory / f"{source_id}_{signature}.npz"
            tmp_path = path.with_suffix(".npz.tmp")
            metadata = {
                "type": "tiles",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tile_size": list(value.tile_size) if value.tile_size is not None else None,
                "boxes": [list(box) for box, _ in value.tiles],
            }
            try:
                arrays: Dict[str, NDArray] = {
                    f"tile_{index}": np.array(tile, copy=True)
                    for index, (_, tile) in enumerate(value.tiles)
                }
                arrays["metadata"] = np.array(json.dumps(metadata))
                with tmp_path.open("wb") as handle:
                    np.savez(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, path)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.debug(
                    "Failed to persist tiled pipeline cache entry",
                    extra={"path": str(path), "error": str(exc)},
                )
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
            return

        array = value
        path = directory / f"{source_id}_{signature}.npy"
        tmp_path = path.with_suffix(".npy.tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.save(handle, array)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug(
                "Failed to persist pipeline cache entry",
                extra={"path": str(path), "error": str(exc)},
            )
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

    def _load_disk_cache(self, source_id: str, signature: str) -> Optional[CacheValue]:
        directory = self._cache_directory
        if directory is None:
            return None
        array_path = directory / f"{source_id}_{signature}.npy"
        if array_path.exists():
            try:
                return np.load(array_path, allow_pickle=False)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.debug(
                    "Failed to load pipeline cache entry",
                    extra={"path": str(array_path), "error": str(exc)},
                )
        tiles_path = directory / f"{source_id}_{signature}.npz"
        if not tiles_path.exists():
            return None
        try:
            with np.load(tiles_path, allow_pickle=False) as data:
                metadata_raw = data.get("metadata")
                if metadata_raw is None:
                    return None
                if isinstance(metadata_raw, np.ndarray):
                    try:
                        metadata_text = metadata_raw.item()
                    except ValueError:
                        metadata_text = str(metadata_raw)
                else:
                    metadata_text = str(metadata_raw)
                metadata = json.loads(str(metadata_text))
                if metadata.get("type") != "tiles":
                    return None
                boxes = [tuple(map(int, box)) for box in metadata.get("boxes", [])]
                tiles: List[Tuple[TileBox, np.ndarray]] = []
                for index, box in enumerate(boxes):
                    tile_key = f"tile_{index}"
                    if tile_key not in data:
                        continue
                    tiles.append((box, np.array(data[tile_key])))
                dtype = np.dtype(metadata.get("dtype", "float32"))
                shape = tuple(int(value) for value in metadata.get("shape", []))
                tile_size_meta = metadata.get("tile_size")
                tile_size = tuple(int(v) for v in tile_size_meta) if tile_size_meta else None
                return TileCacheEntry.from_tiles(shape, dtype, tiles, tile_size=tile_size)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug(
                "Failed to load tiled pipeline cache entry",
                extra={"path": str(tiles_path), "error": str(exc)},
            )
        return None

    def _remove_disk_cache(self, source_id: str) -> None:
        directory = self._cache_directory
        if directory is None:
            return
        prefix = f"{source_id}_"
        patterns = (
            f"{prefix}*.npy",
            f"{prefix}*.npy.tmp",
            f"{prefix}*.npz",
            f"{prefix}*.npz.tmp",
        )
        for pattern in patterns:
            for path in directory.glob(pattern):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()


__all__ = [
    "PipelineCache",
    "PipelineCacheResult",
    "PipelineCacheTileUpdate",
    "StepRecord",
    "TileCacheEntry",
]

