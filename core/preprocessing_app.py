"""Application entry point helpers for preprocessing."""
from __future__ import annotations

import sys
import logging

from PyQt5 import QtWidgets

from ui.preprocessing import MainWindow


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


__all__ = ["main"]
