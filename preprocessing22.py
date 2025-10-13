#!/usr/bin/env python3
"""Entry point for the image preprocessing application."""
from __future__ import annotations

import sys

from core.app_core import AppConfiguration, AppCore
from core.preprocessing import Config
from core.preprocessing_app import main


if __name__ == "__main__":
    app_core = AppCore(
        AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
        )
    )
    sys.exit(main(app_core))
