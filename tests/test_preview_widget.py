import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PyQt5 = pytest.importorskip("PyQt5")
QtCore = PyQt5.QtCore

from yam_processor.ui import PreviewWidget, TiledImageLevel, TiledImageRecord

np = pytest.importorskip("numpy")


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
    assert widget._image_buffer is not None
    assert widget._image_buffer.shape == high.shape


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
