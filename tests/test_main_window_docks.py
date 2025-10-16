import os
from typing import List, Optional

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PyQt5 = pytest.importorskip("PyQt5")
QtCore = PyQt5.QtCore
QtGui = PyQt5.QtGui
QtWidgets = PyQt5.QtWidgets
QtTest = PyQt5.QtTest

from yam_processor.ui.main_window import MainWindow


def _create_main_window(qtbot: "QtBot") -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    QtWidgets.QApplication.processEvents()
    return window


def _assert_actions_sequence(
    actions: List[QtWidgets.QAction], expected: List[Optional[QtWidgets.QAction]]
) -> None:
    assert len(actions) == len(expected)
    for actual, expected_action in zip(actions, expected):
        if expected_action is None:
            assert actual.isSeparator()
        else:
            assert actual is expected_action


def test_main_window_dock_toggle_actions_and_slots(qtbot) -> None:
    window = _create_main_window(qtbot)

    pipeline_spy = QtTest.QSignalSpy(window.pipelineDockVisibilityChanged)
    window.pipeline_dock.setVisible(True)
    QtWidgets.QApplication.processEvents()
    pipeline_spy.clear()

    window.pipeline_toggle_action.trigger()
    QtWidgets.QApplication.processEvents()
    assert len(pipeline_spy) == 1
    assert pipeline_spy[-1][0] is False
    assert not window.pipeline_dock.isVisible()

    window.toggle_pipeline_dock(True)
    QtWidgets.QApplication.processEvents()
    assert len(pipeline_spy) == 2
    assert pipeline_spy[-1][0] is True
    assert window.pipeline_dock.isVisible()

    preview_spy = QtTest.QSignalSpy(window.previewDockVisibilityChanged)
    window.preview_dock.setVisible(True)
    QtWidgets.QApplication.processEvents()
    preview_spy.clear()

    window.preview_toggle_action.trigger()
    QtWidgets.QApplication.processEvents()
    assert len(preview_spy) == 1
    assert preview_spy[-1][0] is False
    assert not window.preview_dock.isVisible()

    window.toggle_preview_dock(True)
    QtWidgets.QApplication.processEvents()
    assert len(preview_spy) == 2
    assert preview_spy[-1][0] is True
    assert window.preview_dock.isVisible()

    diagnostics_spy = QtTest.QSignalSpy(window.diagnosticsDockVisibilityChanged)
    window.diagnostics_dock.setVisible(True)
    QtWidgets.QApplication.processEvents()
    diagnostics_spy.clear()

    window.diagnostics_toggle_action.trigger()
    QtWidgets.QApplication.processEvents()
    assert len(diagnostics_spy) == 1
    assert diagnostics_spy[-1][0] is False
    assert not window.diagnostics_dock.isVisible()

    window.toggle_diagnostics_dock(True)
    QtWidgets.QApplication.processEvents()
    assert len(diagnostics_spy) == 2
    assert diagnostics_spy[-1][0] is True
    assert window.diagnostics_dock.isVisible()


def test_main_window_dock_context_menus(qtbot, monkeypatch) -> None:
    window = _create_main_window(qtbot)

    captured_menus: List[List[QtWidgets.QAction]] = []

    def _capture_exec(self, *args, **kwargs):
        captured_menus.append(list(self.actions()))
        return None

    if hasattr(QtWidgets.QMenu, "exec_"):
        monkeypatch.setattr(QtWidgets.QMenu, "exec_", _capture_exec)
    if hasattr(QtWidgets.QMenu, "exec"):
        monkeypatch.setattr(QtWidgets.QMenu, "exec", _capture_exec)

    pipeline_widget = window.pipeline_dock.widget()
    preview_widget = window.preview_dock.widget()
    diagnostics_widget = window.diagnostics_dock.widget()

    for widget in (pipeline_widget, preview_widget, diagnostics_widget):
        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse,
            QtCore.QPoint(),
            QtCore.QPoint(),
        )
        QtWidgets.QApplication.sendEvent(widget, event)

    assert len(captured_menus) == 3

    pipeline_actions = pipeline_widget.actions()
    _assert_actions_sequence(
        pipeline_actions,
        [
            window.open_project_action,
            window.save_project_action,
            window.save_project_as_action,
            None,
            window.undo_action,
            window.redo_action,
        ],
    )
    assert captured_menus[0] == pipeline_actions

    preview_actions = preview_widget.actions()
    _assert_actions_sequence(
        preview_actions,
        [
            window.pipeline_toggle_action,
            window.preview_toggle_action,
            window.diagnostics_toggle_action,
        ],
    )
    assert captured_menus[1] == preview_actions

    diagnostics_actions = diagnostics_widget.actions()
    _assert_actions_sequence(
        diagnostics_actions,
        [
            window.pipeline_toggle_action,
            window.preview_toggle_action,
            window.diagnostics_toggle_action,
            None,
            window.focus_diagnostics_action,
            window.clear_diagnostics_action,
        ],
    )
    assert captured_menus[2] == diagnostics_actions
