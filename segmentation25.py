#!/usr/bin/env python3
"""Entry point for the segmentation application."""
from __future__ import annotations

import sys
from typing import Sequence

from core.app_core import AppConfiguration, AppCore
from core.segmentation import Config
from core.segmentation_app import main as run_segmentation_app


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv  # CLI parameters are configured via the startup dialog.
    app_core = AppCore(
        AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
        )
    )
    app_core.ensure_bootstrapped()
    _ = app_core.settings
    _ = app_core.autosave
    return run_segmentation_app(app_core)


if __name__ == "__main__":
    sys.exit(main())
