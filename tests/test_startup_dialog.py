"""Tests covering the StartupDialog workflow and launcher integration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import pytest

QtCore = pytest.importorskip("PyQt5.QtCore", exc_type=ImportError)
QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)

from core.app_core import AppConfiguration
from core.application_launcher import (
    StageApplicationSpec,
    StagePaneFactoryResult,
    launch_stage_applications,
)
from core.settings import SettingsManager
from plugins.module_base import ModuleStage
from ui.startup import StartupDialog, StartupModuleOption


def _module_options() -> list[StartupModuleOption]:
    return [
        StartupModuleOption(
            ModuleStage.PREPROCESSING,
            "Preprocessing",
            "Prepare imagery",
            enabled_by_default=True,
        ),
        StartupModuleOption(
            ModuleStage.SEGMENTATION,
            "Segmentation",
            "Detect regions",
            enabled_by_default=True,
        ),
        StartupModuleOption(
            ModuleStage.ANALYSIS,
            "Extraction",
            "Measure features",
            enabled_by_default=False,
        ),
    ]


def test_startup_dialog_returns_selected_stages(qtbot) -> None:
    settings = SettingsManager("Org", "App", seed_defaults=False)
    dialog = StartupDialog(
        _module_options(), diagnostics_enabled=True, settings=settings
    )
    qtbot.addWidget(dialog)

    # Select the segmentation and extraction environments and disable diagnostics.
    list_widget = dialog.findChild(QtWidgets.QListWidget)
    assert list_widget is not None
    assert list_widget.count() == 3

    list_widget.item(0).setCheckState(QtCore.Qt.Unchecked)
    list_widget.item(1).setCheckState(QtCore.Qt.Checked)
    list_widget.item(2).setCheckState(QtCore.Qt.Checked)

    diagnostics_checkbox = dialog.findChild(QtWidgets.QCheckBox)
    assert diagnostics_checkbox is not None
    diagnostics_checkbox.setChecked(False)

    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    assert dialog.selected_stages == (
        ModuleStage.SEGMENTATION,
        ModuleStage.ANALYSIS,
    )
    assert dialog.selected_stage == ModuleStage.SEGMENTATION
    assert dialog.diagnostics_enabled is False

    stored = settings.get("ui/startup/selected_stages")
    assert stored
    assert json.loads(stored) == [
        ModuleStage.SEGMENTATION.name,
        ModuleStage.ANALYSIS.name,
    ]


def test_startup_dialog_disables_accept_without_options(qtbot) -> None:
    dialog = StartupDialog([], diagnostics_enabled=False)
    qtbot.addWidget(dialog)

    button_box = dialog.findChild(QtWidgets.QDialogButtonBox)
    assert button_box is not None
    ok_button = button_box.button(QtWidgets.QDialogButtonBox.Ok)
    assert ok_button is not None
    assert ok_button.isEnabled() is False

    dialog.accept()
    assert dialog.selected_stages == ()


def test_launcher_uses_selected_stage(monkeypatch) -> None:
    import core.application_launcher as launcher

    created_configs: List[AppConfiguration] = []
    created_cores: List[_FakeAppCore] = []

    @dataclass
    class _FakeAppCore:
        config: AppConfiguration

        def __post_init__(self) -> None:
            self.bootstrapped = False
            self.diagnostics_calls: List[tuple[bool, bool]] = []
            self.shutdown_called = False
            created_configs.append(self.config)
            created_cores.append(self)
            self.settings_manager = SettingsManager("Org", "App", seed_defaults=False)
            self.log_handler = type("_Handler", (), {"formatter": None})()
            self.thread_controller = None

        def ensure_bootstrapped(self) -> None:
            self.bootstrapped = True

        @property
        def settings(self) -> SettingsManager:
            return self.settings_manager

        def set_diagnostics_enabled(self, enabled: bool, *, persist: bool = True) -> None:
            self.diagnostics_calls.append((enabled, persist))

        def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(launcher, "AppCore", _FakeAppCore)

    class _FakeWindow:
        def __init__(self, _: _FakeAppCore) -> None:
            self.shown = False
            self.added: List[tuple[ModuleStage, str]] = []

        def add_stage_pane(
            self,
            stage: ModuleStage,
            widget: QtWidgets.QWidget,
            title: str,
            *,
            toolbars=None,
            docks=None,
            status_message=None,
        ) -> None:
            self.added.append((stage, title))

        def show(self) -> None:
            self.shown = True

    windows: List[_FakeWindow] = []

    def window_factory(core: _FakeAppCore) -> _FakeWindow:
        window = _FakeWindow(core)
        windows.append(window)
        return window

    class _DummyPane(QtWidgets.QWidget):
        def on_activated(self) -> None:
            pass

        def on_deactivated(self) -> None:
            pass

        def load_image(self) -> None:
            pass

        def save_outputs(self) -> None:
            pass

        def update_pipeline_summary(self) -> None:
            pass

        def set_diagnostics_visible(self, visible: bool) -> None:
            pass

        def teardown(self) -> None:
            pass

    spec = StageApplicationSpec(
        stage=ModuleStage.SEGMENTATION,
        title="Segmentation",
        description="Detect regions",
        pane_factory=lambda _core, _host: StagePaneFactoryResult(_DummyPane()),
    )

    class _FakeDialog:
        def __init__(
            self,
            options,
            diagnostics_enabled: bool = False,
            settings: SettingsManager | None = None,
        ) -> None:
            self.options = options
            self._diagnostics_enabled = True
            self._selected_stages = [ModuleStage.SEGMENTATION]
            self.settings = settings

        def exec_(self) -> int:
            return QtWidgets.QDialog.Accepted

        @property
        def selected_stages(self):
            return tuple(self._selected_stages)

        @property
        def diagnostics_enabled(self) -> bool:
            return self._diagnostics_enabled

    class _FakeApplication:
        def __init__(self) -> None:
            self.properties: Dict[str, object] = {}
            self.exec_calls = 0

        def setProperty(self, key: str, value: object) -> None:
            self.properties[key] = value

        def exec_(self) -> int:
            self.exec_calls += 1
            return 42

    fake_app = _FakeApplication()
    translation_calls: List[tuple[object, object]] = []

    def fake_translation_bootstrapper(app: object, config: object) -> object:
        translation_calls.append((app, config))
        return "translator"

    themed_apps: List[object] = []

    def fake_theme_applier(app: object) -> None:
        themed_apps.append(app)

    exit_code = launch_stage_applications(
        [spec],
        dialog_factory=_FakeDialog,
        application_factory=lambda: fake_app,
        theme_applier=fake_theme_applier,
        translation_bootstrapper=fake_translation_bootstrapper,
        translation_config_factory=lambda _: "translation-config",
        window_factory=window_factory,
        initial_diagnostics=False,
    )

    assert exit_code == 42
    assert fake_app.properties.get("core.translation_loader") == "translator"
    assert themed_apps == [fake_app]
    assert len(created_configs) == 1
    assert created_configs[0].diagnostics_enabled is True
    assert created_cores[0].bootstrapped is True
    assert created_cores[0].diagnostics_calls == [(True, True)]
    assert created_cores[0].shutdown_called is True
    assert windows and windows[0].shown is True
    assert windows[0].added == [(ModuleStage.SEGMENTATION, "Segmentation")]
    assert translation_calls == [(fake_app, "translation-config")]
