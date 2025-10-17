"""Application entry helpers for extraction."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt5 import QtWidgets

from core.app_core import AppConfiguration, AppCore
from core.extraction import Config
from core.i18n import TranslationConfig, bootstrap_translations
from ui.extraction import MainWindow
from ui.startup import StartupDialog
from ui.theme import apply_application_theme
from plugins.module_base import ModuleStage


def main(app_core: Optional[AppCore] = None) -> int:
    if app_core is None:
        configuration = AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
        )
        app_core = AppCore(configuration)

    app_core.ensure_bootstrapped()

    app = QtWidgets.QApplication(sys.argv)
    translation_config = TranslationConfig(
        directories=app_core.config.translation_directories,
        locales=app_core.config.translation_locales,
        file_prefix=app_core.config.translation_prefix,
    )
    translation_loader = bootstrap_translations(app, translation_config)
    app.setProperty("core.translation_loader", translation_loader)
    apply_application_theme(app)
    dialog = StartupDialog(app_core, ModuleStage.ANALYSIS)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        app_core.shutdown()
        return 0
    window = MainWindow(app_core)
    window.show()
    exit_code = app.exec_()

    app_core.shutdown()

    return exit_code


__all__ = ["main"]
