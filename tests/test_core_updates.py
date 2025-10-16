import json
import threading
from pathlib import Path

import pytest

PyQt5 = pytest.importorskip("PyQt5")

from core.app_core import AppConfiguration, AppCore
from core.thread_controller import ThreadController


class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status = 200
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _create_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppCore:
    config = AppConfiguration(
        diagnostics_enabled=False,
        enable_update_checks=False,
        update_endpoint="https://example.com/update.json",
        plugin_packages=(),
        module_paths=(),
    )
    core = AppCore(config)
    controller = ThreadController()
    core.thread_controller = controller
    monkeypatch.setattr(AppCore, "_current_version", staticmethod(lambda: "1.0.0"))
    return core


def test_check_for_updates_pauses_until_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "version": "2.0.0",
        "notes": "Bug fixes and improvements",
        "release_notes": {"url": "https://example.com/changelog"},
        "download_url": "https://example.com/download",
    }

    def _fake_urlopen(url: str, timeout: float = 10.0) -> _DummyResponse:
        assert url == "https://example.com/update.json"
        return _DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    core = _create_core(tmp_path, monkeypatch)
    controller = core.thread_controller
    assert controller is not None

    core.check_for_updates()

    assert controller.is_paused() is True
    assert core.update_dispatcher.has_pending_update() is True

    executed = threading.Event()

    def _worker(cancel_event: threading.Event, progress) -> str:
        executed.set()
        return "done"

    runner = threading.Thread(target=lambda: controller.run_task(_worker))
    runner.start()

    try:
        assert executed.wait(0.2) is False

        core.acknowledge_update()
        runner.join(timeout=2.0)

        assert executed.wait(1.0) is True
        assert controller.is_paused() is False
        assert core.update_dispatcher.has_pending_update() is False
    finally:
        controller.resume()
        controller.shutdown()


def test_update_check_failure_does_not_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_urlopen(url: str, timeout: float = 10.0):
        raise urllib.error.URLError("offline")

    import urllib.error  # local import for monkeypatch type reference

    monkeypatch.setattr("urllib.request.urlopen", _failing_urlopen)

    core = _create_core(tmp_path, monkeypatch)
    controller = core.thread_controller
    assert controller is not None

    core.check_for_updates()

    assert controller.is_paused() is False
    assert core.update_dispatcher.has_pending_update() is False
    controller.shutdown()


def test_thread_controller_pause_blocks_execution() -> None:
    controller = ThreadController()
    controller.pause()

    executed = threading.Event()

    def _worker(cancel_event: threading.Event, progress) -> str:
        executed.set()
        return "done"

    runner = threading.Thread(target=lambda: controller.run_task(_worker))
    runner.start()

    try:
        assert executed.wait(0.2) is False
        controller.resume()
        runner.join(timeout=2.0)
        assert executed.wait(1.0) is True
    finally:
        controller.resume()
        controller.shutdown()
