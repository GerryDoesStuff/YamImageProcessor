"""Module discovery utilities for plugins."""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Type, TypeVar

from yam_processor.plugins.base import ModuleBase, PipelineStage


LOGGER = logging.getLogger(__name__)


ModuleTypeVar = TypeVar("ModuleTypeVar", bound=ModuleBase)


@dataclass
class ModuleRegistry:
    """In-memory registry of pipeline modules declared by plugins."""

    _modules: Dict[PipelineStage, List[Type[ModuleBase]]] = field(
        default_factory=lambda: {stage: [] for stage in PipelineStage}
    )

    def register(self, module_cls: Type[ModuleTypeVar]) -> None:
        """Validate and store ``module_cls`` for later lookup."""

        if not isinstance(module_cls, type):
            raise TypeError("Registered objects must be classes deriving from ModuleBase.")

        if not issubclass(module_cls, ModuleBase):
            raise TypeError(
                f"{module_cls!r} does not inherit from ModuleBase and cannot be registered."
            )

        stage = getattr(module_cls, "stage", None)
        if not isinstance(stage, PipelineStage):
            raise ValueError(
                f"Module {module_cls.__name__} must declare a PipelineStage via its base class."
            )

        if module_cls in self._modules[stage]:
            LOGGER.debug(
                "Ignoring duplicate registration for module %s (stage=%s)",
                module_cls.__name__,
                stage.value,
            )
            return

        self._modules[stage].append(module_cls)
        LOGGER.info(
            "Registered module %s for stage %s",
            module_cls.__name__,
            stage.value,
        )

    def iter_modules(self, stage: PipelineStage | None = None) -> Iterator[Type[ModuleBase]]:
        """Yield registered modules, optionally filtered by ``stage``."""

        if stage is None:
            for modules in self._modules.values():
                yield from modules
            return

        yield from tuple(self._modules.get(stage, ()))

    def get_modules(self, stage: PipelineStage) -> Tuple[Type[ModuleBase], ...]:
        """Return the registered modules for ``stage`` as an immutable tuple."""

        return tuple(self._modules.get(stage, ()))


class ModuleLoader:
    """Discovers and imports plugin modules from configured packages."""

    def __init__(self, packages: Sequence[str]) -> None:
        self.packages = list(packages)
        self._logger = LOGGER

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
