import types

import pytest

PyQt5 = pytest.importorskip("PyQt5")
QtCore = PyQt5.QtCore
QtWidgets = PyQt5.QtWidgets
QtGui = PyQt5.QtGui

from core.app_core import UpdateDispatcher, UpdateMetadata
from ui.preprocessing import MainWindow, UpdateNotificationDialog


def test_update_notification_dialog_displays_notes(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = UpdateMetadata(
        version="3.0.0",
        notes="Stability improvements",
        release_notes_url="https://example.com/changelog",
        download_url="https://example.com/app",
    )

    dialog = UpdateNotificationDialog(metadata)
    qtbot.addWidget(dialog)

    assert "Stability improvements" in dialog.notes_browser.toPlainText()
    assert dialog.release_notes_link is not None
    assert dialog.download_button is not None

    opened: list[str] = []

    def _fake_open(url: QtCore.QUrl) -> bool:
        opened.append(url.toString())
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", _fake_open)

    assert dialog.download_button is not None
    dialog.download_button.click()
    assert opened == ["https://example.com/app"]


def test_main_window_acknowledges_update_after_dialog(qtbot) -> None:
    dispatcher = UpdateDispatcher(lambda: None)

    window = MainWindow.__new__(MainWindow)
    QtWidgets.QMainWindow.__init__(window)
    window.app_core = types.SimpleNamespace(update_dispatcher=dispatcher)
    MainWindow._init_update_notifications(window)
    qtbot.addWidget(window)

    events: list[str] = []

    class _RecordingDialog(QtWidgets.QDialog):
        def __init__(self, metadata: UpdateMetadata, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.metadata = metadata

        def exec_(self) -> int:
            events.append("dialog")
            return QtWidgets.QDialog.Accepted

    window._update_dialog_factory = lambda metadata: _RecordingDialog(metadata, window)

    original_ack = window.acknowledge_available_update

    def _record_ack() -> None:
        events.append("ack")
        original_ack()

    window.acknowledge_available_update = _record_ack  # type: ignore[assignment]

    metadata = UpdateMetadata(version="4.0.0", notes="Fixes", release_notes_url=None, download_url=None)
    dispatcher.notify(metadata)

    assert events == ["dialog", "ack"]
    assert window._pending_update is None
    assert dispatcher.has_pending_update() is False
