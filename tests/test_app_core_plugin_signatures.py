"""Tests for plugin signature verification during discovery."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable
import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

numpy_stub = types.ModuleType("numpy")
numpy_stub.array = lambda data, copy=None: data  # type: ignore[assignment]
sys.modules.setdefault("numpy", numpy_stub)

cv2_module = types.ModuleType("cv2")
cv2_module.IMREAD_UNCHANGED = -1
cv2_module.imwrite = lambda path, image: True  # type: ignore[assignment]
cv2_module.imread = lambda path, flag=-1: None  # type: ignore[assignment]
sys.modules.setdefault("cv2", cv2_module)

pyqt5_module = types.ModuleType("PyQt5")
qtcore_module = types.ModuleType("PyQt5.QtCore")
qtwidgets_module = types.ModuleType("PyQt5.QtWidgets")
qtgui_module = types.ModuleType("PyQt5.QtGui")
qtcore_module.QObject = type("QObject", (), {})


class _DummySignal:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self._slots: list = []

    def connect(self, slot):  # type: ignore[no-untyped-def]
        self._slots.append(slot)

    def emit(self, *args: object, **kwargs: object) -> None:
        for slot in list(self._slots):
            slot(*args, **kwargs)


def _dummy_pyqt_signal(*args: object, **kwargs: object) -> _DummySignal:
    return _DummySignal()


def _dummy_pyqt_slot(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    def decorator(func):
        return func

    return decorator


class _DummyQRunnable:
    def __init__(self) -> None:
        self._auto_delete = False

    def setAutoDelete(self, value: bool) -> None:
        self._auto_delete = bool(value)

    def run(self) -> None:  # pragma: no cover - not exercised in tests
        pass


class _DummyThreadPool:
    def __init__(self) -> None:
        self._max_threads: int | None = None

    def setMaxThreadCount(self, value: int) -> None:
        self._max_threads = int(value)

    def start(self, runnable: _DummyQRunnable) -> None:
        runnable.run()

    def waitForDone(self) -> None:
        pass


_THREAD_POOL = _DummyThreadPool()


class _DummyQThreadPool:
    @staticmethod
    def globalInstance() -> _DummyThreadPool:
        return _THREAD_POOL


qtcore_module.pyqtSignal = _dummy_pyqt_signal
qtcore_module.pyqtSlot = _dummy_pyqt_slot
qtcore_module.QRunnable = _DummyQRunnable
qtcore_module.QThreadPool = _DummyQThreadPool

sys.modules.setdefault("PyQt5", pyqt5_module)
sys.modules.setdefault("PyQt5.QtCore", qtcore_module)
sys.modules.setdefault("PyQt5.QtWidgets", qtwidgets_module)
sys.modules.setdefault("PyQt5.QtGui", qtgui_module)

from core.app_core import AppConfiguration, AppCore
from core.signing import signature_path_for

from tests.test_module_loader import (  # type: ignore[import-not-found]
    RSAKeyPair,
    _generate_rsa_keypair,
    _sign_module,
)


def _write_signature(module_path: Path, key_pair: RSAKeyPair) -> None:
    signature = _sign_module(module_path.read_bytes(), key_pair)
    signature_path = signature_path_for(module_path)
    signature_path.write_bytes(signature)


def _ensure_module_cleanup(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _prepare_package(
    root: Path,
    package: str,
    key_pair: RSAKeyPair,
    *,
    sign_root: bool = True,
    sign_modules: Iterable[str] = (),
    invalid_signature: bool = False,
) -> None:
    package_dir = root / package
    package_dir.mkdir()

    init_path = package_dir / "__init__.py"
    init_path.write_text(
        """
from typing import Any


def register_module(app: Any) -> None:
    markers = getattr(app, "_loaded_plugins", [])
    markers.append("root")
    app._loaded_plugins = markers
""".strip()
        + "\n",
        encoding="utf-8",
    )
    if sign_root:
        _write_signature(init_path, key_pair)

    for module_name in sign_modules:
        module_path = package_dir / f"{module_name}.py"
        module_path.write_text(
            f"""
from typing import Any


def register_module(app: Any) -> None:
    markers = getattr(app, "_loaded_plugins", [])
    markers.append("{module_name}")
    app._loaded_plugins = markers
""".strip()
            + "\n",
            encoding="utf-8",
        )
        if invalid_signature:
            key_size = (key_pair.modulus.bit_length() + 7) // 8
            signature_path = signature_path_for(module_path)
            signature_path.write_bytes(b"\x00" * key_size)
        else:
            _write_signature(module_path, key_pair)


def _build_app(tmp_path: Path, trust_store: Path, package: str) -> AppCore:
    config = AppConfiguration(
        plugin_packages=[package],
        plugin_trust_store_paths=[trust_store],
    )
    return AppCore(config)


def _create_trust_store(tmp_path: Path, key_pair: RSAKeyPair) -> Path:
    trust_store_path = tmp_path / "trusted.pem"
    trust_store_path.write_text(key_pair.public_pem, encoding="utf-8")
    return trust_store_path


def test_signed_modules_are_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_pair = _generate_rsa_keypair()
    trust_store = _create_trust_store(tmp_path, key_pair)
    package = "signed_plugins"

    _prepare_package(
        tmp_path,
        package,
        key_pair,
        sign_modules=["signed_module"],
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    _ensure_module_cleanup(monkeypatch, package)

    app = _build_app(tmp_path, trust_store, package)
    app._discover_plugins()

    loaded = getattr(app, "_loaded_plugins", [])
    assert loaded == ["root", "signed_module"]
    assert {module.__name__ for module in app.plugins} == {
        package,
        f"{package}.signed_module",
    }
    _ensure_module_cleanup(monkeypatch, package)


def test_missing_signature_skips_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_pair = _generate_rsa_keypair()
    trust_store = _create_trust_store(tmp_path, key_pair)
    package = "unsigned_plugin"

    _prepare_package(
        tmp_path,
        package,
        key_pair,
        sign_modules=["unsigned_module"],
    )

    # Remove the signature to simulate missing file
    module_path = tmp_path / package / "unsigned_module.py"
    signature_path_for(module_path).unlink()

    monkeypatch.syspath_prepend(str(tmp_path))
    _ensure_module_cleanup(monkeypatch, package)

    app = _build_app(tmp_path, trust_store, package)
    app._discover_plugins()

    loaded = getattr(app, "_loaded_plugins", [])
    assert loaded == ["root"]
    assert {module.__name__ for module in app.plugins} == {package}
    _ensure_module_cleanup(monkeypatch, package)


def test_invalid_signature_rejects_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_pair = _generate_rsa_keypair()
    trust_store = _create_trust_store(tmp_path, key_pair)
    package = "tampered_plugin"

    _prepare_package(
        tmp_path,
        package,
        key_pair,
        sign_modules=["tampered_module"],
        invalid_signature=True,
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    _ensure_module_cleanup(monkeypatch, package)

    app = _build_app(tmp_path, trust_store, package)
    app._discover_plugins()

    loaded = getattr(app, "_loaded_plugins", [])
    assert loaded == ["root"]
    assert {module.__name__ for module in app.plugins} == {package}
    _ensure_module_cleanup(monkeypatch, package)

