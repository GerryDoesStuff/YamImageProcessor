"""Compatibility entry point delegating to the unified launcher."""
from __future__ import annotations

from core.application_launcher import default_stage_specifications, launch_stage_applications


def main() -> int:
    return launch_stage_applications(default_stage_specifications())


__all__ = ["main"]
