#!/usr/bin/env python3
"""Entry point for the image preprocessing application."""
from __future__ import annotations

import argparse
import sys

from core.app_core import AppConfiguration, AppCore
from core.preprocessing import Config
from core.preprocessing_app import main


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the preprocessing UI")
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
    sys.exit(main(app_core))
