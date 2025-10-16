from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)

pytest.importorskip("cv2")

from core.thread_controller import OperationCancelled, ThreadController
from processing.pipeline_cache import PipelineCacheResult, PipelineCacheTileUpdate, StepRecord
from processing.pipeline_manager import PipelineManager, PipelineStep
from ui.preprocessing import MainWindow


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
        self._pipeline_manager = PipelineManager([step])
        self.settings = SimpleNamespace(
            snapshot=lambda prefix="": {},
            get=lambda key, default=None: default,
            set=lambda key, value: None,
        )
        self.io_manager = SimpleNamespace(
            save_image=lambda *args, **kwargs: SimpleNamespace(image_path="out.png")
        )

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


@pytest.fixture
def window_factory(qtbot):
    created: list[MainWindow] = []

    def _create(*, stream_threshold: int, tile_size: Tuple[int, int] = (64, 64), delay: float = 0.05):
        cache = FakePipelineCache(
            tile_size=tile_size,
            stream_threshold=stream_threshold,
            delay=delay,
        )
        app_core = FakeAppCore(cache)
        window = MainWindow(app_core)
        qtbot.addWidget(window)
        window.show()
        created.append(window)
        return window, cache, app_core

    yield _create

    for window in created:
        window.thread_controller.shutdown()
        window.close()


def _prime_window_for_image(window: MainWindow, image: np.ndarray) -> str:
    window.original_image = np.array(image, copy=True)
    window.base_image = np.array(image, copy=True)
    window.committed_image = np.array(image, copy=True)
    window.current_preview = np.array(image, copy=True)
    window._set_original_display_image(image)
    window._set_preview_display_image(image)
    source_id = window.pipeline_cache.register_source(image)
    window._source_id = source_id
    window._committed_signature = source_id
    window._preview_signature = source_id
    return source_id


def test_progressive_frames_stream_while_thread_active(qtbot, window_factory) -> None:
    window, cache, _ = window_factory(stream_threshold=64, tile_size=(64, 64), delay=0.05)
    large = np.zeros((256, 256), dtype=np.uint8)
    _prime_window_for_image(window, large)

    window.update_preview()

    qtbot.waitUntil(window.thread_controller.is_running, timeout=1000)

    final_sum = int(np.clip(large + cache.delta, 0, 255).sum())

    def _has_partial_frame() -> bool:
        array = window.preview_display.current_array()
        if array is None:
            return False
        current_sum = int(array.sum())
        return 0 < current_sum < final_sum

    qtbot.waitUntil(_has_partial_frame, timeout=4000)
    assert window.thread_controller.is_running()
    assert cache.last_incremental_count > 0

    qtbot.waitUntil(lambda: not window.thread_controller.is_running(), timeout=5000)
    final = window.preview_display.current_array()
    assert final is not None
    np.testing.assert_array_equal(final, np.clip(large + cache.delta, 0, 255))


def test_cancellation_restores_previous_preview(qtbot, window_factory) -> None:
    window, cache, _ = window_factory(stream_threshold=64, tile_size=(64, 64), delay=0.05)
    baseline = np.full((128, 128), 12, dtype=np.uint8)
    _prime_window_for_image(window, baseline)

    window.update_preview()
    qtbot.waitUntil(window.thread_controller.is_running, timeout=1000)

    qtbot.waitUntil(lambda: cache.last_incremental_count > 0, timeout=4000)

    window.thread_controller.cancel()

    qtbot.waitUntil(lambda: not window.thread_controller.is_running(), timeout=5000)
    qtbot.waitUntil(lambda: window.preview_display.current_array() is not None, timeout=1000)

    restored = window.preview_display.current_array()
    assert restored is not None
    np.testing.assert_array_equal(restored, baseline)
    assert window._progressive_preview_state is None
    assert window._pending_preview_signature is None
    assert window._active_progressive_generation is None
    assert cache.last_incremental_count > 0
    assert not any(
        isinstance(widget, QtWidgets.QProgressDialog)
        for widget in QtWidgets.QApplication.topLevelWidgets()
    )


def test_small_image_updates_without_streaming(qtbot, window_factory) -> None:
    window, cache, _ = window_factory(stream_threshold=512, tile_size=(64, 64), delay=0.01)
    small = np.zeros((32, 32), dtype=np.uint8)
    _prime_window_for_image(window, small)

    window.update_preview()
    qtbot.waitUntil(lambda: not window.thread_controller.is_running(), timeout=3000)

    final = window.preview_display.current_array()
    assert final is not None
    np.testing.assert_array_equal(final, np.clip(small + cache.delta, 0, 255))
    assert cache.last_incremental_count == 0
