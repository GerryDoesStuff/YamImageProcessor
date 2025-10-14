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
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from core.settings import SettingsManager
from core.thread_controller import OperationCancelled
from processing.pipeline_manager import PipelineStep


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
    image: np.ndarray
    steps: List[StepRecord]
    metadata: Dict[str, Any]


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
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
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
    def register_source(self, image: np.ndarray, *, hint: Optional[str] = None) -> str:
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
        image: np.ndarray,
        steps: Sequence[PipelineStep],
        *,
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[Callable[[int], None]] = None,
    ) -> PipelineCacheResult:
        """Evaluate ``steps`` against ``image`` storing intermediate results."""

        final_signature, records = self.predict(source_id, steps)

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
                result = np.array(cached, copy=True)
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

    def get_cached_image(self, source_id: str, signature: str) -> Optional[np.ndarray]:
        """Return a cached copy of the array for ``signature`` if available."""

        with self._lock:
            cache = self._cache.get(source_id)
            cached = cache.get(signature) if cache else None
            if cached is not None:
                return np.array(cached, copy=True)
        disk_cached = self._load_disk_cache(source_id, signature)
        if disk_cached is None:
            return None
        with self._lock:
            cache = self._cache.setdefault(source_id, {})
            cache[signature] = np.array(disk_cached, copy=True)
        return np.array(disk_cached, copy=True)

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

    def _write_disk_cache(self, source_id: str, signature: str, array: np.ndarray) -> None:
        directory = self._cache_directory
        if directory is None:
            return
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

    def _load_disk_cache(self, source_id: str, signature: str) -> Optional[np.ndarray]:
        directory = self._cache_directory
        if directory is None:
            return None
        path = directory / f"{source_id}_{signature}.npy"
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug(
                "Failed to load pipeline cache entry",
                extra={"path": str(path), "error": str(exc)},
            )
            return None

    def _remove_disk_cache(self, source_id: str) -> None:
        directory = self._cache_directory
        if directory is None:
            return
        prefix = f"{source_id}_"
        patterns = (f"{prefix}*.npy", f"{prefix}*.npy.tmp")
        for pattern in patterns:
            for path in directory.glob(pattern):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()


__all__ = ["PipelineCache", "PipelineCacheResult", "StepRecord"]

