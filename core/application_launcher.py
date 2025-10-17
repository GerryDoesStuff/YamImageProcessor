"""Unified application launcher for the Yam Image Processor."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from PyQt5 import QtCore, QtWidgets

from core.app_core import AppConfiguration, AppCore
from core.i18n import TranslationConfig, bootstrap_translations
from core.extraction import Config as ExtractionConfig
from core.preprocessing import Config as PreprocessingConfig
from core.segmentation import Config as SegmentationConfig
from plugins.module_base import ModuleStage
from ui.extraction import MainWindow as ExtractionMainWindow
from ui.preprocessing import MainWindow as PreprocessingMainWindow
from ui.segmentation import MainWindow as SegmentationMainWindow
from ui.startup import StartupDialog, StartupModuleOption
from ui.theme import apply_application_theme


@dataclass(frozen=True)
class StageApplicationSpec:
    """Describe how to bootstrap a specific processing stage."""

    stage: ModuleStage
    title: str
    description: str = ""
    configuration_factory: Callable[[], AppConfiguration] = field(repr=False)
    window_factory: Callable[[AppCore], QtWidgets.QMainWindow] = field(repr=False)


def default_stage_specifications() -> list[StageApplicationSpec]:
    """Return the default set of stage launch specifications."""

    return [
        StageApplicationSpec(
            stage=ModuleStage.PREPROCESSING,
            title="Preprocessing",
            description="Prepare imagery before segmentation or feature extraction.",
            configuration_factory=lambda: AppConfiguration(
                organization=PreprocessingConfig.SETTINGS_ORG,
                application=PreprocessingConfig.SETTINGS_APP,
            ),
            window_factory=lambda core: PreprocessingMainWindow(core),
        ),
        StageApplicationSpec(
            stage=ModuleStage.SEGMENTATION,
            title="Segmentation",
            description="Isolate meaningful regions from the prepared imagery.",
            configuration_factory=lambda: AppConfiguration(
                organization=SegmentationConfig.SETTINGS_ORG,
                application=SegmentationConfig.SETTINGS_APP,
            ),
            window_factory=lambda core: SegmentationMainWindow(core),
        ),
        StageApplicationSpec(
            stage=ModuleStage.ANALYSIS,
            title="Feature Extraction",
            description="Extract quantitative descriptors from segmented data.",
            configuration_factory=lambda: AppConfiguration(
                organization=ExtractionConfig.SETTINGS_ORG,
                application=ExtractionConfig.SETTINGS_APP,
            ),
            window_factory=lambda core: ExtractionMainWindow(core),
        ),
    ]


def launch_stage_applications(
    stage_specs: Sequence[StageApplicationSpec],
    *,
    dialog_factory: Callable[..., StartupDialog] = StartupDialog,
    application_factory: Callable[[], QtWidgets.QApplication] | None = None,
    theme_applier: Callable[[QtWidgets.QApplication], None] = apply_application_theme,
    translation_bootstrapper: Callable[
        [QtWidgets.QApplication, TranslationConfig], object
    ] = bootstrap_translations,
    translation_config_factory: Callable[[AppCore], TranslationConfig] | None = None,
    initial_diagnostics: bool = False,
) -> int:
    """Launch the application shell using ``stage_specs`` for configuration."""

    if not stage_specs:
        raise ValueError("At least one stage specification must be provided.")

    if application_factory is None:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = application_factory()

    options = [
        StartupModuleOption(stage=spec.stage, title=spec.title, description=spec.description)
        for spec in stage_specs
    ]
    dialog = dialog_factory(options, diagnostics_enabled=initial_diagnostics)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return 0

    selected_stage = dialog.selected_stage
    if selected_stage is None:
        return 0

    spec_lookup: Mapping[ModuleStage, StageApplicationSpec] = {
        spec.stage: spec for spec in stage_specs
    }
    stage_spec = spec_lookup[selected_stage]

    configuration = stage_spec.configuration_factory()
    configuration.diagnostics_enabled = dialog.diagnostics_enabled

    app_core = AppCore(configuration)
    app_core.ensure_bootstrapped()
    app_core.set_diagnostics_enabled(dialog.diagnostics_enabled, persist=True)

    translation_config_factory = translation_config_factory or (
        lambda core: TranslationConfig(
            directories=core.config.translation_directories,
            locales=core.config.translation_locales,
            file_prefix=core.config.translation_prefix,
        )
    )
    translation_config = translation_config_factory(app_core)
    translation_loader = translation_bootstrapper(app, translation_config)
    app.setProperty("core.translation_loader", translation_loader)

    theme_applier(app)

    window = stage_spec.window_factory(app_core)
    window.show()

    try:
        exit_code = app.exec_()
    finally:
        app_core.shutdown()

    return exit_code


def main() -> int:
    """Entry point launching the unified processing shell."""

    return launch_stage_applications(default_stage_specifications())


__all__ = [
    "StageApplicationSpec",
    "default_stage_specifications",
    "launch_stage_applications",
    "main",
]
