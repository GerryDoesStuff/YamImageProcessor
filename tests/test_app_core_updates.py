import json
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure PyQt stubs from telemetry tests are registered for imports
from tests import test_app_core_telemetry as _telemetry_stubs  # noqa: F401

from yam_processor.core.app_core import AppConfiguration, AppCore


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
    log_dir = tmp_path / "logs"
    autosave_dir = tmp_path / "autosave"
    session_parent = tmp_path / "session"
    log_dir.mkdir(parents=True, exist_ok=True)
    autosave_dir.mkdir(parents=True, exist_ok=True)
    session_parent.mkdir(parents=True, exist_ok=True)

    config = AppConfiguration(
        developer_diagnostics=False,
        telemetry_opt_in=False,
        log_directory=log_dir,
        autosave_directory=autosave_dir,
        session_temp_parent=session_parent,
        plugin_packages=(),
        module_paths=(),
        enable_update_checks=True,
        update_endpoint="https://example.com/update.json",
    )
    core = AppCore(config)

    monkeypatch.setattr(AppCore, "_current_version", staticmethod(lambda: "1.0.0"))
    monkeypatch.setattr(core, "_init_persistence", lambda: None)
    monkeypatch.setattr(core, "_discover_plugins", lambda: None)
    monkeypatch.setattr(core, "_init_translations", lambda: None)
    monkeypatch.setattr(core, "_configure_user_paths", lambda: None)

    return core


def test_update_check_pauses_background_until_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "version": "2.0.0",
        "release_notes": {"text": "Bug fixes", "url": "https://example.com/changelog"},
        "download_url": "https://example.com/app",
    }

    def _fake_urlopen(url: str, timeout: float = 10.0) -> _DummyResponse:
        assert url == "https://example.com/update.json"
        return _DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    core = _create_core(tmp_path, monkeypatch)
    notifications: list[object] = []

    def _record_notification(metadata):
        notifications.append(metadata)

    core.update_dispatcher.add_listener(_record_notification)

    core.bootstrap()

    try:
        thread_controller = core.thread_controller
        assert thread_controller is not None
        assert thread_controller.is_paused() is True
        assert len(notifications) == 1
        metadata = notifications[0]
        assert metadata.version == "2.0.0"
        assert metadata.notes == "Bug fixes"
        assert metadata.release_notes_url == "https://example.com/changelog"
        assert metadata.download_url == "https://example.com/app"
        assert core.update_dispatcher.has_pending_update() is True

        executed = threading.Event()

        def _worker() -> str:
            executed.set()
            return "done"

        future = thread_controller.submit(_worker)
        assert executed.wait(0.2) is False

        core.update_dispatcher.acknowledge()

        result = future.result(timeout=2.0)
        assert result == "done"
        assert executed.is_set() is True
        assert thread_controller.is_paused() is False
        assert core.update_dispatcher.has_pending_update() is False
    finally:
        core.shutdown()
