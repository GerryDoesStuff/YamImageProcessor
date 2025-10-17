"""Tests covering the StartupDialog workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)

from plugins.module_base import ModuleBase, ModuleMetadata, ModuleStage
from ui.startup import StartupDialog


class _FakeSettings:
    def __init__(self, initial: Dict[str, bool] | None = None) -> None:
        self._store: Dict[str, bool] = dict(initial or {})

    def contains(self, key: str) -> bool:
        return key in self._store

    def get_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._store.get(key, default))


class _FakeModule(ModuleBase):
    def __init__(self, identifier: str, title: str, stage: ModuleStage, description: str = "") -> None:
        self._metadata = ModuleMetadata(
            identifier=identifier,
            title=title,
            stage=stage,
            description=description,
        )
        super().__init__()

    def _build_metadata(self) -> ModuleMetadata:  # type: ignore[override]
        return self._metadata

    def process(self, image: np.ndarray, **_: object) -> np.ndarray:  # type: ignore[override]
        return image


@dataclass
class _FakeCore:
    modules: Dict[ModuleStage, Iterable[ModuleBase]]
    enabled: Dict[ModuleStage, Dict[str, bool]]
    diagnostics_enabled: bool = False
    settings_seed: Dict[str, bool] | None = None

    def __post_init__(self) -> None:
        self.settings_manager = _FakeSettings(self.settings_seed)
        self.module_calls: list[tuple[ModuleStage, str, bool]] = []
        self.diagnostics_calls: list[tuple[bool, bool]] = []

    def iter_modules(self, stage: ModuleStage):
        return list(self.modules.get(stage, ()))

    def module_enabled(self, stage: ModuleStage, identifier: str) -> bool:
        return self.enabled.get(stage, {}).get(identifier, True)

    def set_module_enabled(
        self, stage: ModuleStage, identifier: str, enabled: bool, *, persist: bool = True
    ) -> None:
        self.module_calls.append((stage, identifier, bool(enabled)))
        self.enabled.setdefault(stage, {})[identifier] = bool(enabled)

    def set_diagnostics_enabled(self, enabled: bool, *, persist: bool = True) -> None:
        self.diagnostics_calls.append((bool(enabled), persist))
        self.diagnostics_enabled = bool(enabled)


@pytest.fixture
def sample_core() -> _FakeCore:
    module_a = _FakeModule("A", "Alpha", ModuleStage.PREPROCESSING, "First module")
    module_b = _FakeModule("B", "Beta", ModuleStage.PREPROCESSING, "Second module")
    return _FakeCore(
        modules={ModuleStage.PREPROCESSING: (module_a, module_b)},
        enabled={ModuleStage.PREPROCESSING: {"A": True, "B": False}},
        diagnostics_enabled=False,
        settings_seed={"diagnostics/enabled": True},
    )


def test_startup_dialog_persists_selections(qtbot, sample_core: _FakeCore) -> None:
    dialog = StartupDialog(sample_core, ModuleStage.PREPROCESSING)
    qtbot.addWidget(dialog)

    alpha_checkbox = dialog.findChild(QtWidgets.QCheckBox, None)
    assert alpha_checkbox is not None

    # Toggle the modules and diagnostics selections.
    for checkbox in dialog.findChildren(QtWidgets.QCheckBox):
        if checkbox is dialog._diagnostics_checkbox:
            checkbox.setChecked(False)
        else:
            checkbox.setChecked(not checkbox.isChecked())

    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.Accepted

    assert sample_core.enabled[ModuleStage.PREPROCESSING]["A"] is False
    assert sample_core.enabled[ModuleStage.PREPROCESSING]["B"] is True
    assert sample_core.diagnostics_enabled is False
    assert len(sample_core.module_calls) == 2
    assert sample_core.diagnostics_calls == [(False, True)]


def test_startup_dialog_cancel_leaves_state(qtbot, sample_core: _FakeCore) -> None:
    dialog = StartupDialog(sample_core, ModuleStage.PREPROCESSING)
    qtbot.addWidget(dialog)

    dialog.reject()
    assert dialog.result() == QtWidgets.QDialog.Rejected
    assert sample_core.enabled[ModuleStage.PREPROCESSING]["A"] is True
    assert sample_core.enabled[ModuleStage.PREPROCESSING]["B"] is False
    assert sample_core.diagnostics_enabled is False
    assert sample_core.module_calls == []
    assert sample_core.diagnostics_calls == []
