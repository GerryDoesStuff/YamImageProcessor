"""Application entry point helpers for preprocessing."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from core.app_core import AppConfiguration, AppCore
from core.preprocessing import Config
from ui.preprocessing import MainWindow
from ui.theme import apply_application_theme


def main(app_core: Optional[AppCore] = None) -> int:
    if app_core is None:
        configuration = AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
        )
        app_core = AppCore(configuration)

    app_core.ensure_bootstrapped()

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    apply_application_theme(app)
    window = MainWindow(app_core)
    window.show()
    exit_code = app.exec_()

    app_core.shutdown()

    return exit_code


__all__ = ["main"]
