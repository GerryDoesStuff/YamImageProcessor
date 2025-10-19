import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtWidgets
except ImportError as exc:  # pragma: no cover - skip when Qt bindings unavailable
    QtWidgets = None  # type: ignore[assignment]
    pytestmark = pytest.mark.skip(reason=f"PyQt5 unavailable: {exc}")
else:
    from plugins.module_base import ModuleStage
    from tests._preprocessing_mocks import FakeAppCore, FakePipelineCache
    from ui import ModulePane
    from ui.unified import UnifiedMainWindow


if QtWidgets is not None:

    class _MenuPane(ModulePane):
        def __init__(self, host: QtWidgets.QMainWindow, label: str, actions: list[str]):
            super().__init__(host)
            self._host = host
            self._label = label
            self._actions = actions

        def on_activated(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def on_deactivated(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def load_image(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def save_outputs(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def update_pipeline_summary(self) -> None:  # pragma: no cover - behaviourless stub
            pass

        def set_diagnostics_visible(self, visible: bool) -> None:  # pragma: no cover
            pass

        def refresh_menus(self) -> None:
            menubar = self._host.menuBar()
            menubar.clear()
            menu = menubar.addMenu(self._label)
            for label in self._actions:
                menu.addAction(label)

        def teardown(self) -> None:  # pragma: no cover - behaviourless stub
            pass


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 unavailable")
def test_menu_bar_reflects_active_stage(qtbot) -> None:
    cache = FakePipelineCache(stream_threshold=128)
    app_core = FakeAppCore(cache)
    window = UnifiedMainWindow(app_core)
    qtbot.addWidget(window)
    window.show()
    QtWidgets.QApplication.processEvents()

    first = _MenuPane(window, "First Menu", ["First Action"])
    second = _MenuPane(window, "Second Menu", ["Alpha", "Beta"])

    window.add_stage_pane(ModuleStage.PREPROCESSING, first, "First")
    window.add_stage_pane(ModuleStage.ANALYSIS, second, "Second")
    QtWidgets.QApplication.processEvents()

    def snapshot() -> list[tuple[str, list[str]]]:
        state: list[tuple[str, list[str]]] = []
        for action in window.menuBar().actions():
            menu = action.menu()
            if menu is None:
                state.append((action.text(), []))
            else:
                state.append((action.text(), [child.text() for child in menu.actions()]))
        return state

    assert snapshot() == [("First Menu", ["First Action"])]

    window._tab_widget.setCurrentIndex(1)
    QtWidgets.QApplication.processEvents()
    assert snapshot() == [("Second Menu", ["Alpha", "Beta"])]

    window._tab_widget.setCurrentIndex(0)
    QtWidgets.QApplication.processEvents()
    assert snapshot() == [("First Menu", ["First Action"])]
