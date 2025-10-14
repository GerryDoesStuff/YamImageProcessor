"""Tests covering telemetry enablement constraints in :mod:`app_core`."""

from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
import enum
import sys
import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pyqt5_module = types.ModuleType("PyQt5")
qtcore_module = types.ModuleType("PyQt5.QtCore")
qtwidgets_module = types.ModuleType("PyQt5.QtWidgets")
data_module = types.ModuleType("yam_processor.data")
plugins_base_module = types.ModuleType("yam_processor.plugins.base")
ui_module = types.ModuleType("yam_processor.ui")
ui_module.__path__ = []  # type: ignore[attr-defined]
error_reporter_module = types.ModuleType("yam_processor.ui.error_reporter")
pil_module = types.ModuleType("PIL")
pil_image_module = types.ModuleType("PIL.Image")


class _DummyQCoreApplication:
    @staticmethod
    def instance() -> None:
        return None


class _DummyQLocale:
    @staticmethod
    def system() -> "_DummyQLocale":
        return _DummyQLocale()

    def uiLanguages(self) -> list[str]:
        return []

    def name(self) -> str:
        return ""


class _DummyQTranslator:
    pass


qtcore_module.QCoreApplication = _DummyQCoreApplication
qtcore_module.QLocale = _DummyQLocale
qtcore_module.QTranslator = _DummyQTranslator
qtcore_module.QObject = type("QObject", (), {})
qtcore_module.pyqtSignal = lambda *args, **kwargs: lambda *a, **k: None
pyqt5_module.QtCore = qtcore_module
qtwidgets_module.QWidget = type("QWidget", (), {})


class _DummyQApplication:
    @staticmethod
    def instance() -> None:
        return None


qtwidgets_module.QApplication = _DummyQApplication
pyqt5_module.QtWidgets = qtwidgets_module
sys.modules.setdefault("PyQt5", pyqt5_module)
sys.modules.setdefault("PyQt5.QtCore", qtcore_module)
sys.modules.setdefault("PyQt5.QtWidgets", qtwidgets_module)
data_module.configure_allowed_roots = lambda roots: None
data_module.sanitize_user_path = lambda path, **_: Path(path)
plugins_base_module.PipelineStage = enum.Enum("PipelineStage", {"DUMMY": "dummy"})
plugins_base_module.ModuleBase = type("ModuleBase", (), {"stage": plugins_base_module.PipelineStage.DUMMY})
error_reporter_module.ErrorResolution = enum.Enum("ErrorResolution", {"DISMISS": "dismiss"})
error_reporter_module.present_error_report = (
    lambda *args, **kwargs: error_reporter_module.ErrorResolution.DISMISS
)
ui_module.error_reporter = error_reporter_module

_STUB_MODULES = {
    "yam_processor.data": data_module,
    "yam_processor.plugins.base": plugins_base_module,
    "yam_processor.ui": ui_module,
    "yam_processor.ui.error_reporter": error_reporter_module,
}


class _StubLoader(importlib.abc.Loader):
    def __init__(self, module: types.ModuleType) -> None:
        self._module = module

    def create_module(self, spec):  # type: ignore[override]
        return self._module

    def exec_module(self, module):  # type: ignore[override]
        return None


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[override]
        module = _STUB_MODULES.get(fullname)
        if module is None:
            return None
        loader = _StubLoader(module)
        is_package = hasattr(module, "__path__")
        return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)


sys.meta_path.insert(0, _StubFinder())
sys.modules.update({name: module for name, module in _STUB_MODULES.items() if name not in sys.modules})
class _DummyImage:
    format = "PNG"
    mode = "RGB"
    size = (0, 0)
    info: dict[str, object] = {}

    def __enter__(self) -> "_DummyImage":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def getexif(self) -> dict[str, object]:
        return {}


pil_image_module.open = lambda path: _DummyImage()
pil_image_module.Image = _DummyImage
pil_module.Image = pil_image_module
sys.modules.setdefault("PIL", pil_module)
sys.modules.setdefault("PIL.Image", pil_image_module)

from yam_processor.core.app_core import AppConfiguration, AppCore


def _bootstrap_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    developer_diagnostics: bool,
    telemetry_opt_in: bool,
) -> AppCore:
    """Helper to bootstrap ``AppCore`` with light-weight dependencies."""

    log_dir = tmp_path / "logs"
    autosave_dir = tmp_path / "autosave"
    session_parent = tmp_path / "session"
    log_dir.mkdir(parents=True, exist_ok=True)
    autosave_dir.mkdir(parents=True, exist_ok=True)
    session_parent.mkdir(parents=True, exist_ok=True)

    config = AppConfiguration(
        developer_diagnostics=developer_diagnostics,
        telemetry_opt_in=telemetry_opt_in,
        log_directory=log_dir,
        autosave_directory=autosave_dir,
        session_temp_parent=session_parent,
        plugin_packages=(),
        module_paths=(),
    )
    core = AppCore(config)

    monkeypatch.setattr(core, "_init_persistence", lambda: None)
    monkeypatch.setattr(core, "_init_threading", lambda: None)
    monkeypatch.setattr(core, "_discover_plugins", lambda: None)

    core.bootstrap()
    return core


def test_telemetry_forced_off_without_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core = _bootstrap_core(
        tmp_path,
        monkeypatch,
        developer_diagnostics=False,
        telemetry_opt_in=True,
    )

    try:
        assert core.telemetry_enabled is False
        assert core.settings is not None
        key = core.telemetry_setting_key
        assert core.settings.get(key) is False
    finally:
        core.shutdown()


def test_telemetry_respects_opt_in_with_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core = _bootstrap_core(
        tmp_path,
        monkeypatch,
        developer_diagnostics=True,
        telemetry_opt_in=True,
    )

    try:
        assert core.telemetry_enabled is True
        assert core.settings is not None
        key = core.telemetry_setting_key
        assert core.settings.get(key) is True
    finally:
        core.shutdown()
