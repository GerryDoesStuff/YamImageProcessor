"""Core application modules for the Yam image tools."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "AppConfiguration",
    "AppCore",
    "AutosaveManager",
    "RecoveryManager",
    "AutosaveSnapshot",
    "CrashMarker",
    "RecoverySummary",
    "SettingsManager",
]

_MODULE_MAP: Dict[str, Tuple[str, str]] = {
    "AppConfiguration": ("core.app_core", "AppConfiguration"),
    "AppCore": ("core.app_core", "AppCore"),
    "AutosaveManager": ("core.persistence", "AutosaveManager"),
    "RecoveryManager": ("core.recovery", "RecoveryManager"),
    "AutosaveSnapshot": ("core.recovery", "AutosaveSnapshot"),
    "CrashMarker": ("core.recovery", "CrashMarker"),
    "RecoverySummary": ("core.recovery", "RecoverySummary"),
    "SettingsManager": ("core.settings", "SettingsManager"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    try:
        module_name, attribute = _MODULE_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(f"module 'core' has no attribute '{name}'") from exc
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper
    return sorted(set(globals()) | set(__all__))
