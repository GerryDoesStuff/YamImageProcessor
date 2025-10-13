"""Caching utilities for pipeline step results and reproducibility metadata."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        self._settings = settings
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._lock = threading.Lock()
        self._load_metadata()

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
            cache[source_id] = np.array(array, copy=True)
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

        return source_id

    def discard_cache(self, source_id: str) -> None:
        """Remove in-memory cache entries for ``source_id`` (metadata retained)."""

        with self._lock:
            self._cache.pop(source_id, None)

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
                    cache[record.signature] = np.array(result, copy=True)
            if progress is not None:
                progress(int(((index + 1) / total_steps) * 100))

        if not records:
            with self._lock:
                cache[final_signature] = np.array(result, copy=True)

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
            if not cache:
                return None
            cached = cache.get(signature)
            if cached is None:
                return None
            return np.array(cached, copy=True)

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
            return
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
                    key: dict(value) for key, value in entries.items() if isinstance(value, MutableMapping)
                }
        self._metadata = data

    def _persist_metadata_locked(self) -> None:
        if self._settings is None:
            return
        serialisable = json.dumps(self._metadata, sort_keys=True)
        self._settings.set(self.SETTINGS_KEY, serialisable)


__all__ = ["PipelineCache", "PipelineCacheResult", "StepRecord"]

