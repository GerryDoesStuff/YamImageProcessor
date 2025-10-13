#!/usr/bin/env python3
"""Entry point for the segmentation application."""
from __future__ import annotations

import argparse
import sys

from core.app_core import AppConfiguration, AppCore
from core.segmentation import Config
from core.segmentation_app import main


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the segmentation UI")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable verbose diagnostics logging and console output.",
    )
    args, remaining = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0], *remaining]
    return args


if __name__ == "__main__":
    arguments = _parse_args(sys.argv[1:])
    app_core = AppCore(
        AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
            diagnostics_enabled=arguments.diagnostics,
        )
    )
    app_core.ensure_bootstrapped()
    _ = app_core.settings
    sys.exit(main(app_core))
