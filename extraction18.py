#!/usr/bin/env python3
"""Entry point for the feature extraction application."""
from __future__ import annotations

import sys
from typing import Sequence

from core.application_launcher import main as run_unified_launcher


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv  # CLI parameters are managed by the startup dialog.
    return run_unified_launcher()


if __name__ == "__main__":
    sys.exit(main())
