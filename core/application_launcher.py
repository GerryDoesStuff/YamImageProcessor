"""Unified application launcher for the Yam Image Processor."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from PyQt5 import QtCore, QtWidgets

from core.app_core import AppConfiguration, AppCore
from core.i18n import TranslationConfig, bootstrap_translations
from plugins.module_base import ModuleStage
from ui.startup import StartupDialog, StartupModuleOption
from ui.theme import apply_application_theme
from ui.unified import UnifiedMainWindow


@dataclass(frozen=True)
class StageApplicationSpec:
    """Describe how to bootstrap a specific processing stage."""

    stage: ModuleStage
    title: str
    pane_factory: Callable[[AppCore], QtWidgets.QWidget]
    description: str = ""
    enabled_by_default: bool = True


def default_stage_specifications() -> list[StageApplicationSpec]:
    """Return the default set of stage launch specifications."""

    return [
        StageApplicationSpec(
            stage=ModuleStage.PREPROCESSING,
            title="Preprocessing",
            description="Prepare imagery before segmentation or feature extraction.",
            pane_factory=_create_preprocessing_pane,
            enabled_by_default=True,
        ),
        StageApplicationSpec(
            stage=ModuleStage.SEGMENTATION,
            title="Segmentation",
            description="Isolate meaningful regions from the prepared imagery.",
            pane_factory=_create_segmentation_pane,
            enabled_by_default=True,
        ),
        StageApplicationSpec(
            stage=ModuleStage.ANALYSIS,
            title="Feature Extraction",
            description="Extract quantitative descriptors from segmented data.",
            pane_factory=_create_extraction_pane,
            enabled_by_default=False,
        ),
    ]


def _create_preprocessing_pane(core: AppCore) -> QtWidgets.QWidget:
    """Import and construct the preprocessing pane lazily."""

    from PyQt5 import QtCore as _QtCore

    from ui.preprocessing import MainWindow as PreprocessingMainWindow

    window = PreprocessingMainWindow(core)
    window.setWindowFlag(_QtCore.Qt.Widget, True)
    return window


def _create_segmentation_pane(core: AppCore) -> QtWidgets.QWidget:
    """Import and construct the segmentation pane lazily."""

    from PyQt5 import QtCore as _QtCore

    from ui.segmentation import MainWindow as SegmentationMainWindow

    window = SegmentationMainWindow(core)
    window.setWindowFlag(_QtCore.Qt.Widget, True)
    return window


def _create_extraction_pane(core: AppCore) -> QtWidgets.QWidget:
    """Import and construct the extraction pane lazily."""

    from PyQt5 import QtCore as _QtCore

    from ui.extraction import MainWindow as ExtractionMainWindow

    window = ExtractionMainWindow(core)
    window.setWindowFlag(_QtCore.Qt.Widget, True)
    return window


def launch_stage_applications(
    stage_specs: Sequence[StageApplicationSpec],
    *,
    configuration_factory: Callable[[], AppConfiguration] | None = None,
    dialog_factory: Callable[..., StartupDialog] = StartupDialog,
    application_factory: Callable[[], QtWidgets.QApplication] | None = None,
    window_factory: Callable[[AppCore], UnifiedMainWindow] = UnifiedMainWindow,
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

    configuration_factory = configuration_factory or AppConfiguration
    configuration = configuration_factory()
    configuration.diagnostics_enabled = bool(initial_diagnostics)

    app_core = AppCore(configuration)
    app_core.ensure_bootstrapped()

    options = [
        StartupModuleOption(
            stage=spec.stage,
            title=spec.title,
            description=spec.description,
            enabled_by_default=spec.enabled_by_default,
        )
        for spec in stage_specs
    ]

    dialog = dialog_factory(
        options,
        diagnostics_enabled=initial_diagnostics,
        settings=app_core.settings,
    )
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        app_core.shutdown()
        return 0

    selected_stages = tuple(dialog.selected_stages)
    if not selected_stages:
        app_core.shutdown()
        return 0

    spec_lookup: Mapping[ModuleStage, StageApplicationSpec] = {
        spec.stage: spec for spec in stage_specs
    }

    diagnostics_requested = dialog.diagnostics_enabled
    configuration.diagnostics_enabled = diagnostics_requested
    app_core.set_diagnostics_enabled(diagnostics_requested, persist=True)

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

    window = window_factory(app_core)
    panes_added = 0
    for stage in selected_stages:
        stage_spec = spec_lookup.get(stage)
        if stage_spec is None:
            continue
        pane = stage_spec.pane_factory(app_core)
        window.add_stage_pane(stage, pane, stage_spec.title)
        panes_added += 1

    if panes_added == 0:
        close_method = getattr(window, "close", None)
        if callable(close_method):
            close_method()
        app_core.shutdown()
        return 0
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


if __name__ == "__main__":
    raise SystemExit(main())
