import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PyQt5.QtCore", exc_type=ImportError)
QtGui = pytest.importorskip("PyQt5.QtGui", exc_type=ImportError)
QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)

pytest.importorskip("cv2", exc_type=ImportError)

from processing.pipeline_cache import PipelineCacheTileUpdate
from tests._preprocessing_mocks import FakeAppCore, FakePipelineCache
from ui.preprocessing import MainWindow, ModuleWindow, PreprocessingPane
from yam_processor.ui import PreviewWidget, TiledImageLevel, TiledImageRecord

np = pytest.importorskip("numpy")


@contextmanager
def preprocessing_pane_host(qtbot, *, use_wrapper: bool = False):
    cache = FakePipelineCache()
    app_core = FakeAppCore(cache)
    if use_wrapper:
        host: QtWidgets.QWidget = MainWindow(app_core)
        assert isinstance(host, ModuleWindow)
        pane = host.pane
        widget: QtWidgets.QWidget = host
    else:
        host = QtWidgets.QMainWindow()
        pane = PreprocessingPane(app_core, host=host)
        host.setCentralWidget(pane)
        widget = pane
    qtbot.addWidget(host)
    host.show()
    try:
        yield SimpleNamespace(host=host, pane=pane, cache=cache, app_core=app_core, widget=widget)
    finally:
        pane.thread_controller.shutdown()
        if not use_wrapper:
            pane.teardown()
        host.close()


def test_preview_widget_progressive_loading(qtbot) -> None:
    widget = PreviewWidget()
    qtbot.addWidget(widget)

    low = np.full((32, 32, 3), 64, dtype=np.uint8)
    high = np.full((64, 64, 3), 192, dtype=np.uint8)

    record = TiledImageRecord(
        [
            TiledImageLevel(scale=0.25, fetch=lambda: low.copy()),
            TiledImageLevel(scale=1.0, fetch=lambda: high.copy()),
        ]
    )

    widget.set_image(record)

    qtbot.waitUntil(
        lambda: widget._image_buffer is not None
        and widget._image_buffer.shape == low.shape,
        timeout=1000,
    )
    assert widget._image_buffer is not None
    assert widget._image_buffer.shape == low.shape

    qtbot.waitUntil(
        lambda: widget._image_buffer is not None
        and widget._image_buffer.shape == high.shape,
        timeout=2000,
    )


def test_preview_widget_displays_nd_arrays(qtbot) -> None:
    widget = PreviewWidget()
    qtbot.addWidget(widget)

    cube = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    widget.update_array(cube, dims=("z", "y", "x"))

    qtbot.waitUntil(lambda: widget._image_buffer is not None, timeout=500)
    assert widget._slice_controls.isVisible()
    assert widget._axis_combo.count() == 1
    assert widget._slice_slider.maximum() == 1
    np.testing.assert_array_equal(widget._image_buffer, cube[1])

    widget._slice_slider.setValue(0)
    qtbot.waitUntil(lambda: np.array_equal(widget._image_buffer, cube[0]), timeout=500)


def test_preview_widget_eager_path_for_single_level(qtbot) -> None:
    widget = PreviewWidget()
    qtbot.addWidget(widget)

    eager_called = False

    def fetch() -> np.ndarray:
        nonlocal eager_called
        eager_called = True
        return np.full((32, 32, 3), 255, dtype=np.uint8)

    record = TiledImageRecord([TiledImageLevel(scale=1.0, fetch=fetch)])
    widget.set_image(record)

    assert eager_called
    assert widget._pending_levels == []
    qtbot.waitUntil(lambda: widget._image_buffer is not None, timeout=500)
    assert widget._image_buffer is not None
    assert widget._image_buffer.shape == (32, 32, 3)


def _send_wheel_event(view: QtWidgets.QGraphicsView, delta: int) -> None:
    viewport = view.viewport()
    position = QtCore.QPointF(viewport.rect().center())
    event = QtGui.QWheelEvent(
        position,
        position,
        QtCore.QPoint(0, delta),
        QtCore.QPoint(0, delta),
        delta,
        QtCore.Qt.Vertical,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    QtWidgets.QApplication.sendEvent(viewport, event)


def test_preview_widget_zoom_persists_during_updates(qtbot) -> None:
    widget = PreviewWidget()
    qtbot.addWidget(widget)

    base = np.full((64, 64, 3), 180, dtype=np.uint8)
    record = TiledImageRecord([TiledImageLevel(scale=1.0, fetch=lambda: base.copy())])
    widget.set_image(record)
    qtbot.waitUntil(lambda: widget._image_buffer is not None, timeout=500)

    initial_scale = widget._view.transform().m11()
    _send_wheel_event(widget._view, 120)
    qtbot.waitUntil(lambda: widget._view.transform().m11() > initial_scale, timeout=500)
    zoomed_scale = widget._view.transform().m11()

    widget.update_array(np.full((64, 64, 3), 30, dtype=np.uint8))
    assert pytest.approx(widget._view.transform().m11(), rel=1e-6) == zoomed_scale


def test_progressive_cancel_restores_large_frame(qtbot) -> None:
    with preprocessing_pane_host(qtbot) as ctx:
        pane = ctx.pane
        baseline = np.full((8, 8), 20, dtype=np.uint8)
        pane.current_preview = baseline.copy()
        pane.preview_display.update_array(baseline)
        pane._progressive_previous_frame = baseline.copy()
        pane._pending_preview_signature = "sig"
        pane._active_progressive_generation = 1

        update = PipelineCacheTileUpdate(
            source_id="src",
            final_signature="sig",
            step_signature="sig",
            step_index=1,
            total_steps=1,
            box=(0, 0, 4, 4),
            tile=np.full((4, 4), 200, dtype=np.uint8),
            shape=baseline.shape,
            dtype=np.dtype(np.uint8),
            tile_size=None,
            from_cache=False,
        )
        pane._handle_pipeline_incremental_update(update, 1)
        qtbot.waitUntil(
            lambda: pane.preview_display._image_buffer is not None
            and np.array_equal(
                pane.preview_display._image_buffer[:4, :4], np.full((4, 4), 200)
            ),
            timeout=500,
        )

        pane._restore_progressive_baseline()
        restored = pane.preview_display.current_array()
        assert restored is not None
        assert np.array_equal(restored, baseline)


def test_progressive_cancel_restores_eager_frame(qtbot) -> None:
    with preprocessing_pane_host(qtbot) as ctx:
        pane = ctx.pane
        baseline = np.full((4, 4, 3), 90, dtype=np.uint8)
        pane.current_preview = baseline.copy()
        pane.preview_display.update_array(baseline)
        pane._progressive_previous_frame = baseline.copy()

        pane.preview_display.update_array(np.full((4, 4, 3), 10, dtype=np.uint8))
        pane._restore_progressive_baseline()

        restored = pane.preview_display.current_array()
        assert restored is not None
        assert np.array_equal(restored, baseline)


def test_preview_widget_remains_responsive_with_large_sources(qtbot) -> None:
    widget = PreviewWidget()
    qtbot.addWidget(widget)

    low = np.zeros((16, 16, 3), dtype=np.uint8)
    high = np.zeros((256, 256, 3), dtype=np.uint8)

    started = threading.Event()
    finished = threading.Event()

    def low_fetch() -> np.ndarray:
        return low.copy()

    def high_fetch() -> np.ndarray:
        started.set()
        time.sleep(0.3)
        finished.set()
        return high.copy()

    record = TiledImageRecord(
        [
            TiledImageLevel(scale=0.0625, fetch=low_fetch),
            TiledImageLevel(scale=1.0, fetch=high_fetch),
        ]
    )

    widget.set_image(record)

    qtbot.waitUntil(
        lambda: widget._image_buffer is not None
        and widget._image_buffer.shape == low.shape,
        timeout=1000,
    )

    assert started.wait(timeout=1000)

    timer_triggered: list[bool] = []
    QtCore.QTimer.singleShot(20, lambda: timer_triggered.append(True))
    qtbot.waitUntil(lambda: bool(timer_triggered), timeout=200)
    assert not finished.is_set()

    qtbot.waitUntil(lambda: finished.is_set(), timeout=2000)
    qtbot.waitUntil(
        lambda: widget._image_buffer is not None
        and widget._image_buffer.shape == high.shape,
        timeout=2000,
    )
