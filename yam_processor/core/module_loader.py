"""Module discovery utilities for plugins."""
from __future__ import annotations

import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Iterable, List, Sequence


class ModuleLoader:
    """Discovers and imports plugin modules from configured packages."""

    def __init__(self, packages: Sequence[str]) -> None:
        self.packages = list(packages)
        self._logger = logging.getLogger(__name__)

    def discover(self) -> List[ModuleType]:
        discovered: List[ModuleType] = []
        for package_name in self.packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as exc:  # pragma: no cover - discovery failure path
                self._logger.warning(
                    "Failed to import plugin package", extra={"component": "ModuleLoader", "package": package_name, "error": str(exc)}
                )
                continue

            modules = self._discover_from_package(package)
            discovered.extend(modules)
        return discovered

    def _discover_from_package(self, package: ModuleType) -> Iterable[ModuleType]:
        package_path = getattr(package, "__path__", None)
        if not package_path:
            return []

        modules: List[ModuleType] = []
        for _finder, name, _ in pkgutil.iter_modules(package_path):
            full_name = f"{package.__name__}.{name}"
            try:
                modules.append(importlib.import_module(full_name))
            except Exception as exc:  # pragma: no cover - plugin import guard
                self._logger.error(
                    "Failed to load plugin module",
                    extra={"component": "ModuleLoader", "module": full_name, "error": str(exc)},
                )
        return modules
