"""Application entry helpers for extraction."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt5 import QtWidgets

from core.app_core import AppConfiguration, AppCore
from core.extraction import Config
from ui.extraction import MainWindow


def main(app_core: Optional[AppCore] = None) -> int:
    if app_core is None:
        configuration = AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
        )
        app_core = AppCore(configuration)

    app_core.ensure_bootstrapped()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(app_core)
    window.show()
    exit_code = app.exec_()

    app_core.shutdown()

    return exit_code


__all__ = ["main"]
