from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from core.thread_controller import OperationCancelled, ThreadController
from plugins.module_base import ModuleStage
from processing.pipeline_cache import PipelineCacheResult, PipelineCacheTileUpdate, StepRecord
from processing.pipeline_manager import PipelineManager, PipelineStep


class FakePipelineCache:
    """Lightweight cache that emits deterministic progressive updates."""

    def __init__(
        self,
        *,
        tile_size: Tuple[int, int] = (64, 64),
        stream_threshold: int = 96,
        delta: int = 37,
        delay: float = 0.05,
    ) -> None:
        self.tile_size = tile_size
        self.stream_threshold = stream_threshold
        self.delta = int(delta)
        self.delay = float(delay)
        self._sources: Dict[str, np.ndarray] = {}
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._metadata: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._predictions: Dict[
            Tuple[str, Tuple[Tuple[str, Tuple[Tuple[str, object], ...], bool], ...]],
            Tuple[str, Tuple[StepRecord, ...]],
        ] = {}
        self._signature_counter = 0
        self._lock = threading.Lock()
        self.last_incremental_count = 0

    @staticmethod
    def _step_key(steps: Iterable[PipelineStep]) -> Tuple[Tuple[str, Tuple[Tuple[str, object], ...], bool], ...]:
        entries = []
        for step in steps:
            params = tuple(sorted(step.params.items()))
            entries.append((step.name, params, bool(step.enabled)))
        return tuple(entries)

    def register_source(self, image: np.ndarray, *, hint: Optional[str] = None) -> str:
        with self._lock:
            source_id = f"source::{len(self._sources) + 1}"
            self._sources[source_id] = np.array(image, copy=True)
        return source_id

    def discard_cache(self, source_id: str) -> None:
        with self._lock:
            stale = [key for key in self._cache if key[0] == source_id]
            for key in stale:
                self._cache.pop(key, None)
                self._metadata.pop(key, None)

    def metadata_for(self, source_id: str, signature: str) -> Dict[str, object]:
        with self._lock:
            return dict(self._metadata.get((source_id, signature), {}))

    def get_cached_image(self, source_id: str, signature: str) -> Optional[np.ndarray]:
        with self._lock:
            cached = self._cache.get((source_id, signature))
        if cached is None:
            return None
        return np.array(cached, copy=True)

    def predict(
        self, source_id: str, steps: Iterable[PipelineStep]
    ) -> Tuple[str, Tuple[StepRecord, ...]]:
        key = (source_id, self._step_key(steps))
        with self._lock:
            self._signature_counter += 1
            final_signature = f"{source_id}::sig::{self._signature_counter}"
            records = tuple(
                StepRecord(
                    name=step.name,
                    enabled=bool(step.enabled),
                    params=dict(step.params),
                    signature=f"{final_signature}::step::{index}",
                    index=index,
                )
                for index, step in enumerate(steps, start=1)
            )
            self._predictions[key] = (final_signature, records)
        return final_signature, records

    def compute(
        self,
        *,
        source_id: str,
        image: np.ndarray,
        steps: Iterable[PipelineStep],
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[Callable[[int], None]] = None,
        incremental: Optional[Callable[[PipelineCacheTileUpdate], None]] = None,
    ) -> PipelineCacheResult:
        key = (source_id, self._step_key(steps))
        with self._lock:
            prediction = self._predictions.get(key)
        if prediction is None:
            prediction = self.predict(source_id, steps)
        final_signature, records = prediction

        base = np.array(image, copy=True)
        result = np.clip(base.astype(np.int32) + self.delta, 0, 255).astype(base.dtype)
        height, width = result.shape[:2]
        total_steps = max(1, len(records))
        self.last_incremental_count = 0

        should_stream = max(height, width) > self.stream_threshold
        tiles: list[Tuple[int, int, int, int]] = []
        if should_stream:
            tile_w, tile_h = self.tile_size
            for top in range(0, height, tile_h):
                bottom = min(top + tile_h, height)
                for left in range(0, width, tile_w):
                    right = min(left + tile_w, width)
                    tiles.append((left, top, right, bottom))
        else:
            tiles.append((0, 0, width, height))

        for index, (left, top, right, bottom) in enumerate(tiles, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled()
            tile = np.array(result[top:bottom, left:right], copy=True)
            if incremental is not None and should_stream:
                update = PipelineCacheTileUpdate(
                    source_id=source_id,
                    final_signature=final_signature,
                    step_signature=records[-1].signature if records else final_signature,
                    step_index=total_steps,
                    total_steps=total_steps,
                    box=(left, top, right, bottom),
                    tile=tile,
                    shape=tuple(int(dim) for dim in result.shape),
                    dtype=np.dtype(tile.dtype),
                    tile_size=self.tile_size,
                    from_cache=False,
                )
                incremental(update)
                self.last_incremental_count += 1
            if progress is not None:
                progress(int(index * 100 / len(tiles)))
            time.sleep(self.delay)

        metadata = {"final_signature": final_signature, "delta": self.delta}
        payload = PipelineCacheResult(
            source_id=source_id,
            final_signature=final_signature,
            image=np.array(result, copy=True),
            steps=list(records),
            metadata=dict(metadata),
        )
        with self._lock:
            self._cache[(source_id, final_signature)] = np.array(result, copy=True)
            self._metadata[(source_id, final_signature)] = dict(metadata)
        return payload


class FakeAppCore:
    """Minimal AppCore replacement used for UI tests."""

    def __init__(self, pipeline_cache: FakePipelineCache) -> None:
        self.pipeline_cache = pipeline_cache
        self.thread_controller = ThreadController(max_workers=1)
        self.log_handler = None
        self.update_dispatcher = None
        self.diagnostics_enabled = False
        self.telemetry_opt_in = False
        self.telemetry_enabled = False
        self.recovery_manager = None
        self.autosave = SimpleNamespace(mark_dirty=lambda *args, **kwargs: None)
        step = PipelineStep(
            "mock-step",
            lambda arr: arr,
            enabled=True,
            params={},
            supports_tiled_input=True,
        )
        step.stage = ModuleStage.PREPROCESSING
        self._pipeline_manager = PipelineManager([step])
        self.settings = SimpleNamespace(
            snapshot=lambda prefix="": {},
            get=lambda key, default=None: default,
            set=lambda key, value: None,
        )

    def get_pipeline_manager(self) -> PipelineManager:
        return self._pipeline_manager

    def get_preprocessing_pipeline_manager(self) -> PipelineManager:
        return self._pipeline_manager

    def get_modules(self, stage):  # pragma: no cover - menu construction
        return ()

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self.diagnostics_enabled = enabled

    def configure_telemetry(self, opt_in: bool) -> None:
        self.telemetry_opt_in = opt_in

    def load_preprocessing_pipeline(self, payload: Dict[str, object]) -> None:  # pragma: no cover
        pass


__all__ = ["FakeAppCore", "FakePipelineCache"]
