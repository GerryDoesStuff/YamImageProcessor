"""Lightweight application bootstrap exposing shared services."""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from plugins.module_base import ModuleBase, ModuleStage

from .thread_controller import ThreadController

from .io_manager import IOManager
from .persistence import AutosaveManager
from .settings import SettingsManager

from .logging import init_logging
from processing.pipeline_cache import PipelineCache
from processing.pipeline_manager import PipelineManager, PipelineStep


@dataclass
class AppConfiguration:
    """Configuration values used when constructing :class:`AppCore`."""

    organization: str = "MicroscopicApp"
    application: str = "ImageProcessor"
    log_level: int = logging.INFO
    diagnostics_enabled: bool = False
    log_directory: Path = Path("logs")
    log_filename: str = "application.log"
    plugin_packages: Sequence[str] = field(default_factory=lambda: ["modules"])
    module_paths: Sequence[Path | str] = field(default_factory=list)
    max_workers: Optional[int] = None
    autosave_directory: Optional[Path] = None
    autosave_interval_seconds: float = 120.0
    autosave_backup_retention: int = 5
    autosave_enabled_default: bool = True


class AppCore:
    """Provide access to process-wide services such as settings and threading."""

    def __init__(self, config: Optional[AppConfiguration] = None) -> None:
        self.config = config or AppConfiguration()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)
        self.settings_manager: Optional[SettingsManager] = None
        self.thread_controller: Optional[ThreadController] = None
        self._io_manager: Optional[IOManager] = None
        self.autosave_manager: Optional[AutosaveManager] = None
        self._module_catalog: Dict[ModuleStage, Dict[str, ModuleBase]] = {
            stage: {} for stage in ModuleStage
        }
        self.plugins: List[object] = []
        self._log_handler: Optional[logging.Handler] = None
        self._bootstrapped = False
        self._preprocessing_manager: Optional[PipelineManager] = None
        self._preprocessing_templates: Dict[str, PipelineStep] = {}
        self._pipeline_cache: Optional[PipelineCache] = None

    # ------------------------------------------------------------------
    # Lifecycle management
    def bootstrap(self) -> None:
        """Initialise logging, settings, threading and plugin discovery."""

        if self._bootstrapped:
            return

        self._configure_logging()
        self._init_settings()
        self._init_threading()
        self._discover_plugins()
        self._bootstrapped = True
        self.logger.info(
            "Application core bootstrapped",
            extra={
                "component": "AppCore",
                "plugins": len(self.plugins),
            },
        )

    def shutdown(self) -> None:
        """Release any resources held by the shared services."""

        if not self._bootstrapped:
            return

        if self.thread_controller is not None:
            self.thread_controller.shutdown()
            self.thread_controller = None

        if self.autosave_manager is not None:
            self.autosave_manager.shutdown()
            self.autosave_manager = None

        self.logger.info("Application core shutdown", extra={"component": "AppCore"})
        self._bootstrapped = False

    # ------------------------------------------------------------------
    # Public helpers
    @property
    def qsettings(self):
        """Expose the underlying ``QSettings`` backend."""

        if self.settings_manager is None:
            raise RuntimeError("Settings manager not initialised. Call bootstrap() first.")
        return self.settings_manager.backend

    @property
    def settings(self) -> SettingsManager:
        """Return the high level settings manager."""

        if self.settings_manager is None:
            raise RuntimeError("Settings manager not initialised. Call bootstrap() first.")
        return self.settings_manager

    @property
    def io_manager(self) -> IOManager:
        """Return the shared image persistence helper."""

        if self._io_manager is None:
            raise RuntimeError("IO manager not initialised. Call bootstrap() first.")
        return self._io_manager

    @property
    def autosave(self) -> AutosaveManager:
        """Return the configured autosave manager."""

        if self.autosave_manager is None:
            raise RuntimeError("Autosave manager not initialised. Call bootstrap() first.")
        return self.autosave_manager

    def ensure_bootstrapped(self) -> None:
        """Ensure :meth:`bootstrap` has been executed."""

        if not self._bootstrapped:
            self.bootstrap()

    # ------------------------------------------------------------------
    # Pipeline helpers
    @property
    def pipeline_cache(self) -> PipelineCache:
        if self._pipeline_cache is None:
            if self.settings_manager is None:
                raise RuntimeError("Settings manager not initialised. Call bootstrap() first.")
            self._pipeline_cache = PipelineCache(self.settings_manager)
        return self._pipeline_cache

    def get_preprocessing_pipeline_manager(self) -> PipelineManager:
        if self._preprocessing_manager is None:
            templates: Dict[str, PipelineStep] = {}
            for module in self.iter_modules(ModuleStage.PREPROCESSING):
                step = module.create_pipeline_step()
                templates[step.name] = step
            self._preprocessing_templates = templates
            self._preprocessing_manager = PipelineManager(templates.values())
        return self._preprocessing_manager

    def preprocessing_step_template(self, name: str) -> PipelineStep:
        return self._preprocessing_templates[name].clone()

    def load_preprocessing_pipeline(self, payload: Dict[str, Any]) -> None:
        manager = self.get_preprocessing_pipeline_manager()
        templates = self._preprocessing_templates
        steps: List[PipelineStep] = []
        seen: set[str] = set()
        for entry in payload.get("steps", []):
            name = entry.get("name")
            if not isinstance(name, str) or name not in templates:
                continue
            step = templates[name].clone()
            step.enabled = bool(entry.get("enabled", step.enabled))
            params = entry.get("params", {})
            if isinstance(params, dict):
                step.params.update(params)
            steps.append(step)
            seen.add(name)
        for name, template in templates.items():
            if name in seen:
                continue
            steps.append(template.clone())
        manager.replace_steps(steps, preserve_history=True)

    # ------------------------------------------------------------------
    # Internal helpers
    def _configure_logging(self) -> None:
        self._log_handler = init_logging(
            diagnostics_enabled=self.config.diagnostics_enabled,
            level=self.config.log_level,
            log_directory=self.config.log_directory,
            log_filename=self.config.log_filename,
        )
        self.logger.setLevel(
            logging.DEBUG if self.config.diagnostics_enabled else self.config.log_level
        )

    def _init_settings(self) -> None:
        defaults: Dict[str, Any] = {
            "autosave/interval_seconds": self.config.autosave_interval_seconds,
            "autosave/backup_retention": self.config.autosave_backup_retention,
            "autosave/enabled": self.config.autosave_enabled_default,
        }
        if self.config.autosave_directory is not None:
            defaults["autosave/workspace"] = str(self.config.autosave_directory)

        self.settings_manager = SettingsManager(
            self.config.organization,
            self.config.application,
            defaults=defaults,
        )
        self._pipeline_cache = PipelineCache(self.settings_manager)
        stored_diagnostics = self.settings_manager.get(
            "diagnostics/enabled", self.config.diagnostics_enabled
        )
        self.set_diagnostics_enabled(
            self._coerce_bool(stored_diagnostics), persist=False
        )
        self._io_manager = IOManager(self.settings_manager)
        self._init_autosave()
        self.logger.debug(
            "Settings manager initialised",
            extra={"component": "AppCore"},
        )

    def _init_threading(self) -> None:
        self.thread_controller = ThreadController(max_workers=self.config.max_workers)
        self.logger.debug(
            "Thread controller initialised",
            extra={"component": "AppCore"},
        )

    def _init_autosave(self) -> None:
        if self.settings_manager is None or self._io_manager is None:
            raise RuntimeError("Settings manager and IO manager must be initialised first")

        workspace = self.settings_manager.autosave_workspace()
        if workspace is None:
            workspace = (
                self.config.autosave_directory
                if self.config.autosave_directory is not None
                else Path.home() / f".{self.config.application.lower()}" / "autosave"
            )
            self.settings_manager.set_autosave_workspace(workspace)

        autosave_logger = logging.getLogger(f"{__name__}.Autosave")
        self.autosave_manager = AutosaveManager(
            self.settings_manager,
            self._io_manager,
            autosave_directory=workspace,
            interval_seconds=self.settings_manager.autosave_interval(),
            logger=autosave_logger,
        )
        self.logger.debug(
            "Autosave manager initialised",
            extra={"component": "AppCore", "autosave_dir": str(workspace)},
        )

    def _discover_plugins(self) -> None:
        discovered: Dict[str, ModuleType] = {}
        for package_name in self.config.plugin_packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as exc:  # pragma: no cover - discovery failure path
                self.logger.warning(
                    "Failed to import plugin package",
                    extra={
                        "component": "AppCore",
                        "package": package_name,
                        "error": str(exc),
                    },
                )
                continue

            self._invoke_register(package)
            discovered[package.__name__] = package

            package_path = getattr(package, "__path__", None)
            if not package_path:
                continue

            for _, name, _ in pkgutil.walk_packages(package_path, package.__name__ + "."):
                if name in discovered:
                    continue
                try:
                    module = importlib.import_module(name)
                except Exception as exc:  # pragma: no cover - plugin import guard
                    self.logger.warning(
                        "Failed to load plugin module",
                        extra={
                            "component": "AppCore",
                            "module": name,
                            "error": str(exc),
                        },
                    )
                    continue
                self._invoke_register(module)
                discovered[name] = module

        self.plugins = list(discovered.values())
        self.logger.debug(
            "Plugin discovery complete",
            extra={"component": "AppCore", "count": len(self.plugins)},
        )

    def _invoke_register(self, module: ModuleType) -> None:
        register = getattr(module, "register_module", None)
        if not callable(register):
            return
        try:
            register(self)
            self.logger.debug(
                "Plugin registered",
                extra={"component": "AppCore", "module": module.__name__},
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Plugin registration failed",
                extra={
                    "component": "AppCore",
                    "module": module.__name__,
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Module management helpers
    def register_module(self, module_cls: type[ModuleBase]) -> None:
        """Register ``module_cls`` with the module catalogue."""

        if not isinstance(module_cls, type) or not issubclass(module_cls, ModuleBase):
            raise TypeError("Modules must be registered using ModuleBase subclasses.")

        module = module_cls()
        metadata = module.metadata
        stage_modules = self._module_catalog.setdefault(metadata.stage, {})
        if metadata.identifier in stage_modules:
            self.logger.warning(
                "Duplicate module identifier detected",
                extra={
                    "component": "AppCore",
                    "identifier": metadata.identifier,
                    "stage": metadata.stage.value,
                },
            )
            return

        stage_modules[metadata.identifier] = module
        self.logger.info(
            "Registered module",
            extra={
                "component": "AppCore",
                "identifier": metadata.identifier,
                "stage": metadata.stage.value,
            },
        )

    def iter_modules(self, stage: ModuleStage | None = None) -> Iterator[ModuleBase]:
        """Yield registered modules, optionally filtered by ``stage``."""

        if stage is None:
            for modules in self._module_catalog.values():
                yield from modules.values()
            return
        yield from self._module_catalog.get(stage, {}).values()

    def get_modules(self, stage: ModuleStage) -> Tuple[ModuleBase, ...]:
        """Return the registered modules for ``stage``."""

        return tuple(self._module_catalog.get(stage, {}).values())

    # ------------------------------------------------------------------
    # Diagnostics helpers
    @property
    def diagnostics_enabled(self) -> bool:
        return self.config.diagnostics_enabled

    @property
    def log_handler(self) -> Optional[logging.Handler]:
        return self._log_handler

    def set_diagnostics_enabled(self, enabled: bool, *, persist: bool = True) -> None:
        enabled = bool(enabled)
        self.config.diagnostics_enabled = enabled
        self._log_handler = init_logging(
            diagnostics_enabled=enabled,
            level=self.config.log_level,
            log_directory=self.config.log_directory,
            log_filename=self.config.log_filename,
        )
        self.logger.setLevel(logging.DEBUG if enabled else self.config.log_level)
        if persist and self.settings_manager is not None:
            self.settings_manager.set("diagnostics/enabled", enabled)

    def set_log_level(self, level: int) -> None:
        self.config.log_level = level
        self._log_handler = init_logging(
            diagnostics_enabled=self.config.diagnostics_enabled,
            level=level,
            log_directory=self.config.log_directory,
            log_filename=self.config.log_filename,
        )
        if not self.config.diagnostics_enabled:
            self.logger.setLevel(level)

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)


__all__ = ["AppConfiguration", "AppCore"]
