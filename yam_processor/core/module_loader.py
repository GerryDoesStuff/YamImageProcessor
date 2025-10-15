"""Module discovery utilities for plugins."""
from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
import sys
from dataclasses import dataclass, field
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Type, TypeVar

from yam_processor.plugins.base import ModuleBase, PipelineStage
from yam_processor.core.signing import (
    InvalidSignatureError,
    MissingSignatureError,
    ModuleSignatureVerifier,
    SignatureVerificationError,
    signature_path_for,
)


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
    """Discovers and imports plugin modules from configured packages or paths."""

    _MODULE_NAMESPACE = "yam_processor.modules.dynamic"

    def __init__(
        self,
        packages: Sequence[str] | None = None,
        module_paths: Sequence[Path | str] | None = None,
        signature_verifier: ModuleSignatureVerifier | None = None,
        signature_extension: str = ".sig",
    ) -> None:
        self.packages = list(packages or [])
        self._logger = LOGGER
        default_modules_dir = Path(__file__).resolve().parents[2] / "modules"
        provided_paths = [Path(path) for path in (module_paths or [])]
        if default_modules_dir not in provided_paths:
            provided_paths.append(default_modules_dir)
        self._signature_verifier = signature_verifier
        self._signature_extension = signature_extension
        seen: set[Path] = set()
        self.module_paths: List[Path] = []
        for path in provided_paths:
            if path not in seen:
                seen.add(path)
                self.module_paths.append(path)
        self._module_index = 0

    def discover(self) -> List[ModuleType]:
        discovered: List[ModuleType] = []
        for package_name in self.packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as exc:  # pragma: no cover - discovery failure path
                self._logger.warning(
                    "Failed to import plugin package",
                    extra={
                        "component": "ModuleLoader",
                        "package": package_name,
                        "error": str(exc),
                    },
                )
                continue

            modules = self._discover_from_package(package)
            discovered.extend(modules)
        for path in self.module_paths:
            modules = self._discover_from_path(path)
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
                self._logger.warning(
                    "Failed to load plugin module",
                    extra={
                        "component": "ModuleLoader",
                        "module": full_name,
                        "error": str(exc),
                    },
                )
        return modules

    def _discover_from_path(self, path: Path) -> List[ModuleType]:
        if not path.exists():
            return []

        if path.is_dir():
            modules: List[ModuleType] = []
            for module_file in sorted(path.glob("*.py")):
                module = self._load_module_from_file(module_file)
                if module is not None:
                    modules.append(module)
            return modules

        if path.suffix != ".py":
            return []

        module = self._load_module_from_file(path)
        return [module] if module is not None else []

    def _load_module_from_file(self, file_path: Path) -> ModuleType | None:
        module_name = self._build_module_name(file_path)
        loader = SourceFileLoader(module_name, str(file_path))
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            self._logger.warning(
                "Failed to create module spec from path",
                extra={
                    "component": "ModuleLoader",
                    "module_path": str(file_path),
                },
            )
            return None

        if self._signature_verifier is not None:
            try:
                module_bytes = file_path.read_bytes()
            except OSError as exc:  # pragma: no cover - defensive logging path
                self._logger.warning(
                    "Failed to read module bytes for signature verification",
                    extra={
                        "component": "ModuleLoader",
                        "module_path": str(file_path),
                        "error": str(exc),
                    },
                )
                return None
            try:
                signature_bytes = self._load_signature(file_path)
            except MissingSignatureError as exc:
                self._logger.warning(
                    "Missing signature for module",
                    extra={
                        "component": "ModuleLoader",
                        "module_path": str(file_path),
                        "error": str(exc),
                    },
                )
                return None
            try:
                self._signature_verifier.verify(module_bytes, signature_bytes)
            except InvalidSignatureError as exc:
                self._logger.warning(
                    "Rejected module due to invalid signature",
                    extra={
                        "component": "ModuleLoader",
                        "module_path": str(file_path),
                        "error": str(exc),
                    },
                )
                return None
            except SignatureVerificationError as exc:
                self._logger.warning(
                    "Signature verification failed",
                    extra={
                        "component": "ModuleLoader",
                        "module_path": str(file_path),
                        "error": str(exc),
                    },
                )
                return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - plugin import guard
            self._logger.warning(
                "Failed to load plugin module from path",
                extra={
                    "component": "ModuleLoader",
                    "module_path": str(file_path),
                    "error": str(exc),
                },
            )
            sys.modules.pop(module_name, None)
            return None
        return module

    def _build_module_name(self, file_path: Path) -> str:
        self._module_index += 1
        stem = file_path.stem.replace("-", "_")
        return f"{self._MODULE_NAMESPACE}.{stem}_{self._module_index}"

    def _load_signature(self, module_path: Path) -> bytes:
        signature_path = signature_path_for(module_path, self._signature_extension)
        try:
            return signature_path.read_bytes()
        except FileNotFoundError as exc:
            raise MissingSignatureError(
                f"Signature file not found at {signature_path}"
            ) from exc
        except OSError as exc:  # pragma: no cover - defensive logging path
            raise SignatureVerificationError(
                f"Unable to read signature for module {module_path}: {exc}"
            ) from exc
