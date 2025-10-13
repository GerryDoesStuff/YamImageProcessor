"""Tests for the ModuleLoader utilities."""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

numpy_stub = types.ModuleType("numpy")
numpy_stub.array = lambda data, copy=None: data  # type: ignore[assignment]
sys.modules.setdefault("numpy", numpy_stub)

yam_processor_stub = types.ModuleType("yam_processor")
yam_processor_stub.__path__ = [str(PROJECT_ROOT / "yam_processor")]  # type: ignore[attr-defined]
sys.modules.setdefault("yam_processor", yam_processor_stub)

core_stub = types.ModuleType("yam_processor.core")
core_stub.__path__ = [str(PROJECT_ROOT / "yam_processor" / "core")]  # type: ignore[attr-defined]
sys.modules.setdefault("yam_processor.core", core_stub)

MODULE_LOADER_PATH = PROJECT_ROOT / "yam_processor" / "core" / "module_loader.py"
SPEC = importlib.util.spec_from_file_location(
    "yam_processor.core.module_loader", MODULE_LOADER_PATH
)
assert SPEC and SPEC.loader is not None
module_loader_module = importlib.util.module_from_spec(SPEC)
sys.modules["yam_processor.core.module_loader"] = module_loader_module
SPEC.loader.exec_module(module_loader_module)
ModuleLoader = module_loader_module.ModuleLoader


@pytest.fixture
def module_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample_plugin.py"
    file_path.write_text("VALUE = 42\n", encoding="utf-8")
    return file_path


def test_discover_from_module_file(module_file: Path) -> None:
    loader = ModuleLoader(packages=[], module_paths=[module_file])

    discovered = loader.discover()

    values = [getattr(module, "VALUE", None) for module in discovered]
    assert 42 in values


def test_discover_from_directory_logs_warning_on_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good_file = tmp_path / "good_plugin.py"
    good_file.write_text("FLAG = 'ok'\n", encoding="utf-8")

    bad_file = tmp_path / "bad_plugin.py"
    bad_file.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    caplog.set_level(logging.WARNING)

    loader = ModuleLoader(packages=[], module_paths=[tmp_path])

    discovered = loader.discover()

    assert any(getattr(module, "FLAG", None) == "ok" for module in discovered)

    warning_records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and getattr(record, "module_path", "") == str(bad_file)
    ]
    assert warning_records, "Expected a warning log entry for the failing module import"
