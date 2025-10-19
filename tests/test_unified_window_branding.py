import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - skip when Qt bindings unavailable
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    pytestmark = pytest.mark.skip(reason=f"PyQt5 unavailable: {exc}")
else:
    from plugins.module_base import ModuleStage
    from tests._preprocessing_mocks import FakeAppCore, FakePipelineCache
    from ui import ModulePane
    from ui.unified import UnifiedMainWindow


if QtWidgets is not None:

    class _BrandingClobberPane(ModulePane):
        """Stub pane that attempts to mutate the host window chrome."""

        def __init__(self, host: QtWidgets.QMainWindow) -> None:
            super().__init__(host)
            self._host = host
            host.setWindowTitle("Clobbered Title")
            icon = QtGui.QIcon()
            pixmap = QtGui.QPixmap(16, 16)
            pixmap.fill(QtGui.QColor("red"))
            icon.addPixmap(pixmap)
            host.setWindowIcon(icon)

        def on_activated(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def on_deactivated(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def load_image(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def save_outputs(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def update_pipeline_summary(self) -> None:  # pragma: no cover - stub
            pass

        def set_diagnostics_visible(self, visible: bool) -> None:  # pragma: no cover
            pass

        def teardown(self) -> None:  # pragma: no cover - behaviourless stub
            pass


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 unavailable")
def test_unified_window_restores_branding(qtbot) -> None:
    cache = FakePipelineCache(stream_threshold=128)
    app_core = FakeAppCore(cache)
    window = UnifiedMainWindow(app_core)
    qtbot.addWidget(window)
    window.show()
    QtWidgets.QApplication.processEvents()

    canonical_title = window.windowTitle()
    canonical_icon_key = window.windowIcon().cacheKey()

    first = _BrandingClobberPane(window)
    second = _BrandingClobberPane(window)

    window.add_stage_pane(ModuleStage.PREPROCESSING, first, "First")
    window.add_stage_pane(ModuleStage.ANALYSIS, second, "Second")
    QtWidgets.QApplication.processEvents()

    assert window.windowTitle() == canonical_title
    assert window.windowIcon().cacheKey() == canonical_icon_key

    window._tab_widget.setCurrentIndex(1)
    QtWidgets.QApplication.processEvents()

    assert window.windowTitle() == canonical_title
    assert window.windowIcon().cacheKey() == canonical_icon_key

