#!/usr/bin/env python3
"""Entry point for the image preprocessing application."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from core.app_core import AppConfiguration, AppCore
from core.preprocessing import Config
from core.preprocessing_app import main as run_preprocessing_app


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


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    arguments = _parse_args(list(argv))
    app_core = AppCore(
        AppConfiguration(
            organization=Config.SETTINGS_ORG,
            application=Config.SETTINGS_APP,
            diagnostics_enabled=arguments.diagnostics,
        )
    )
    app_core.ensure_bootstrapped()
    _ = app_core.settings
    _ = app_core.autosave
    return run_preprocessing_app(app_core)


if __name__ == "__main__":
    sys.exit(main())
