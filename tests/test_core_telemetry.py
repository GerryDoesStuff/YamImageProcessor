import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the PyQt and Pillow stubs from the yam_processor telemetry tests
from tests import test_app_core_telemetry as _telemetry_stubs  # noqa: F401

qtcore_module = sys.modules.setdefault("PyQt5.QtCore", types.ModuleType("PyQt5.QtCore"))
if not hasattr(qtcore_module, "QRunnable"):
    class _DummyQRunnable:
        def __init__(self) -> None:
            self._auto_delete = True

        def setAutoDelete(self, value: bool) -> None:  # noqa: N802 - Qt API surface
            self._auto_delete = value

    qtcore_module.QRunnable = _DummyQRunnable

if not hasattr(qtcore_module, "QThreadPool"):
    class _DummyThreadPool:
        def __init__(self) -> None:
            self._max = None

        @staticmethod
        def globalInstance() -> "_DummyThreadPool":  # noqa: N802 - Qt API surface
            return _DummyThreadPool()

        def setMaxThreadCount(self, value: int) -> None:
            self._max = value

        def start(self, runnable: "_DummyQRunnable") -> None:  # noqa: F821
            if hasattr(runnable, "run"):
                runnable.run()

        def waitForDone(self) -> None:
            return None

    qtcore_module.QThreadPool = _DummyThreadPool

if not hasattr(qtcore_module, "pyqtSlot"):
    qtcore_module.pyqtSlot = lambda *args, **kwargs: (lambda func: func)

try:  # pragma: no cover - optional dependency in tests
    import numpy  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object  # type: ignore[attr-defined]
    numpy.float32 = float  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy

if "cv2" not in sys.modules:  # pragma: no cover - optional dependency in tests
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_GRAY2BGR = 0  # type: ignore[attr-defined]
    cv2.COLOR_BGR2RGB = 0  # type: ignore[attr-defined]
    cv2.cvtColor = lambda image, mode: image  # type: ignore[attr-defined]
    sys.modules["cv2"] = cv2

from core.app_core import AppConfiguration, AppCore
from core.settings import SettingsManager


@pytest.fixture
def settings_manager(monkeypatch: pytest.MonkeyPatch) -> SettingsManager:
    monkeypatch.setattr("core.settings.QSettings", None, raising=False)
    return SettingsManager(
        "TestOrg",
        "TelemetryApp",
        defaults={},
        seed_defaults=False,
    )


def test_telemetry_remains_disabled_without_diagnostics(settings_manager: SettingsManager) -> None:
    config = AppConfiguration(diagnostics_enabled=False)
    core = AppCore(config)
    core.settings_manager = settings_manager

    core.configure_telemetry(True)

    assert core.telemetry_opt_in is True
    assert core.telemetry_enabled is False
    assert settings_manager.get(core.telemetry_setting_key) is True

    core.set_diagnostics_enabled(True)

    assert core.telemetry_enabled is True
    assert settings_manager.get(core.telemetry_setting_key) is True


def test_telemetry_setting_round_trip(settings_manager: SettingsManager) -> None:
    config = AppConfiguration(diagnostics_enabled=True)
    core = AppCore(config)
    core.settings_manager = settings_manager

    core.configure_telemetry(True)

    assert core.telemetry_enabled is True
    assert settings_manager.get(core.telemetry_setting_key) is True

    core.configure_telemetry(False)

    assert core.telemetry_enabled is False
    assert settings_manager.get(core.telemetry_setting_key) is False

    # Simulate reading the persisted choice in a new session
    new_core = AppCore(config)
    new_core.settings_manager = settings_manager
    effective = new_core._resolve_telemetry_opt_in()

    assert effective is False
    assert new_core.telemetry_opt_in is False

    settings_manager.set(new_core.telemetry_setting_key, True)
    new_core.config.diagnostics_enabled = True
    effective = new_core._resolve_telemetry_opt_in()

    assert effective is True
    assert new_core.telemetry_opt_in is True
