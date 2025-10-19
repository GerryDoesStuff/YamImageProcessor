from __future__ import annotations

import os
from types import MethodType, SimpleNamespace
from typing import Tuple

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)

pytest.importorskip("cv2")

from plugins.module_base import ModuleStage
from tests._preprocessing_mocks import FakeAppCore, FakePipelineCache
from ui.preprocessing import MainWindow, ModuleWindow, PreprocessingPane
from ui.unified import UnifiedMainWindow


@pytest.fixture
def window_factory(qtbot):
    created: list[tuple[QtWidgets.QWidget, PreprocessingPane, bool]] = []

    def _create(
        *,
        stream_threshold: int,
        tile_size: Tuple[int, int] = (64, 64),
        delay: float = 0.05,
        use_wrapper: bool = True,
    ) -> SimpleNamespace:
        cache = FakePipelineCache(
            tile_size=tile_size,
            stream_threshold=stream_threshold,
            delay=delay,
        )
        app_core = FakeAppCore(cache)
        if use_wrapper:
            host: QtWidgets.QWidget = ModuleWindow(app_core)
            pane = host.pane
            window = host
        else:
            host = QtWidgets.QMainWindow()
            pane = PreprocessingPane(app_core, host=host)
            host.setCentralWidget(pane)
            window = host
        qtbot.addWidget(host)
        host.show()
        created.append((host, pane, use_wrapper))
        return SimpleNamespace(window=window, pane=pane, cache=cache, app_core=app_core)

    yield _create

    for host, pane, use_wrapper in created:
        pane.thread_controller.shutdown()
        if not use_wrapper:
            pane.teardown()
        host.close()


def _coerce_pane(target: object) -> PreprocessingPane:
    if isinstance(target, PreprocessingPane):
        return target
    if isinstance(target, ModuleWindow):
        return target.pane
    pane = getattr(target, "pane", None)
    if isinstance(pane, PreprocessingPane):
        return pane
    raise TypeError(f"Unsupported target for pane extraction: {type(target)!r}")


def _prime_window_for_image(window: object, image: np.ndarray) -> str:
    pane = _coerce_pane(window)
    pane.original_image = np.array(image, copy=True)
    pane.base_image = np.array(image, copy=True)
    pane.committed_image = np.array(image, copy=True)
    pane.current_preview = np.array(image, copy=True)
    pane._set_original_display_image(image)
    pane._set_preview_display_image(image)
    source_id = pane.pipeline_cache.register_source(image)
    pane._source_id = source_id
    pane._committed_signature = source_id
    pane._preview_signature = source_id
    return source_id


def test_progressive_frames_stream_while_thread_active(qtbot, window_factory) -> None:
    bundle = window_factory(stream_threshold=64, tile_size=(64, 64), delay=0.05)
    window = bundle.window
    cache = bundle.cache
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
    bundle = window_factory(stream_threshold=64, tile_size=(64, 64), delay=0.05)
    window = bundle.window
    cache = bundle.cache
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
    bundle = window_factory(stream_threshold=512, tile_size=(64, 64), delay=0.01)
    window = bundle.window
    cache = bundle.cache
    small = np.zeros((32, 32), dtype=np.uint8)
    _prime_window_for_image(window, small)

    window.update_preview()
    qtbot.waitUntil(lambda: not window.thread_controller.is_running(), timeout=3000)

    final = window.preview_display.current_array()
    assert final is not None
    np.testing.assert_array_equal(final, np.clip(small + cache.delta, 0, 255))
    assert cache.last_incremental_count == 0


def test_preprocessing_pane_embeds_in_unified_window(qtbot) -> None:
    cache = FakePipelineCache(stream_threshold=128, tile_size=(64, 64), delay=0.01)
    app_core = FakeAppCore(cache)
    unified = UnifiedMainWindow(app_core)
    qtbot.addWidget(unified)
    unified.show()

    pane = PreprocessingPane(app_core, host=unified)

    activated: list[bool] = []

    def _on_activated(self: PreprocessingPane) -> None:
        activated.append(True)
        PreprocessingPane.on_activated(self)

    pane.on_activated = MethodType(_on_activated, pane)

    load_calls: list[bool] = []
    save_calls: list[bool] = []

    def _load_image(self: PreprocessingPane) -> None:
        load_calls.append(True)

    def _save_outputs(self: PreprocessingPane) -> None:
        save_calls.append(True)

    pane.load_image = MethodType(_load_image, pane)
    pane.save_outputs = MethodType(_save_outputs, pane)

    unified.add_stage_pane(
        ModuleStage.PREPROCESSING,
        pane,
        title="Preprocessing",
    )

    tab_widget = unified.findChild(QtWidgets.QTabWidget, "stageTabWidget")
    assert tab_widget is not None
    assert tab_widget.widget(0) is pane

    qtbot.waitUntil(lambda: bool(activated), timeout=1000)

    unified.load_image()
    assert load_calls

    unified.save_outputs()
    assert save_calls

    pane.thread_controller.shutdown()
    pane.teardown()
    unified.close()
