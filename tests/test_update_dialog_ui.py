from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

from yam_processor.core.app_core import UpdateMetadata
from yam_processor.ui.main_window import MainWindow
from yam_processor.ui.update_dialog import UpdateDialog


def test_update_dialog_displays_release_notes(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = UpdateMetadata(
        version="2.5.0",
        notes="Stability improvements",
        release_notes_url="https://example.com/changelog",
        download_url="https://example.com/app",
    )

    dialog = UpdateDialog(metadata)
    qtbot.addWidget(dialog)

    assert dialog.notes_browser.toPlainText() == "Stability improvements"
    assert dialog.install_button.isEnabled() is True
    assert dialog.release_notes_link is not None
    assert "changelog" in dialog.release_notes_link.text()

    opened_urls: list[str] = []

    def _fake_open(url: QtCore.QUrl) -> bool:
        opened_urls.append(url.toString())
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", _fake_open)

    dialog.install_button.click()
    assert dialog.selected_action == UpdateDialog.Action.INSTALL.value
    assert opened_urls == ["https://example.com/app"]


class _DummyDispatcher:
    def __init__(self) -> None:
        self._listener = None
        self.acknowledged = 0

    def add_listener(self, callback) -> None:
        self._listener = callback

    def remove_listener(self, callback) -> None:
        if self._listener == callback:
            self._listener = None

    def acknowledge(self) -> None:
        self.acknowledged += 1


class _SignalStub:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)


def test_main_window_waits_for_dialog_acknowledgement(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MainWindow, "screenChanged", _SignalStub(), raising=False)
    window = MainWindow()
    qtbot.addWidget(window)

    dispatcher = _DummyDispatcher()
    window.set_update_dispatcher(dispatcher)

    dialogs: list[QtWidgets.QDialog] = []

    class _RecordingDialog(QtWidgets.QDialog):
        def __init__(self, metadata: UpdateMetadata, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.metadata = metadata
            self.exec_called = False

        def exec_(self) -> int:  # pragma: no cover - exercised in tests
            self.exec_called = True
            return QtWidgets.QDialog.Accepted

    def _factory(metadata: UpdateMetadata) -> QtWidgets.QDialog:
        dialog = _RecordingDialog(metadata, window)
        dialogs.append(dialog)
        return dialog

    window._update_dialog_factory = _factory  # type: ignore[attr-defined]

    acknowledgement_order: list[str] = []

    def _acknowledge() -> None:
        acknowledgement_order.append("after" if dialogs[0].exec_called else "before")

    monkeypatch.setattr(window, "acknowledge_available_update", _acknowledge)

    metadata = UpdateMetadata(version="3.0.0", notes="Fixes", download_url=None, release_notes_url=None)
    window._on_update_available(metadata)

    assert acknowledgement_order == ["after"]
    assert len(dialogs) == 1
    assert dialogs[0].metadata is metadata


def test_main_window_acknowledges_dispatcher_after_dialog(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MainWindow, "screenChanged", _SignalStub(), raising=False)
    window = MainWindow()
    qtbot.addWidget(window)

    dispatcher = _DummyDispatcher()
    window.set_update_dispatcher(dispatcher)

    class _AutoAcceptDialog(QtWidgets.QDialog):
        def __init__(self, metadata: UpdateMetadata, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.metadata = metadata

        def exec_(self) -> int:  # pragma: no cover - deterministic return
            return QtWidgets.QDialog.Accepted

    window._update_dialog_factory = lambda metadata: _AutoAcceptDialog(metadata, window)  # type: ignore[attr-defined]

    metadata = UpdateMetadata(version="3.1.0", notes=None, release_notes_url=None, download_url=None)
    window._on_update_available(metadata)

    assert dispatcher.acknowledged == 1
    assert window._pending_update is None
