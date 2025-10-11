"""Application core bootstrap handling logging, settings, plugins and threading."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .logging_config import LoggingConfigurator, LoggingOptions
from .module_loader import ModuleLoader, ModuleRegistry
from .settings_manager import SettingsManager
from .threading import ThreadController


@dataclass
class AppConfiguration:
    """Configuration for the application bootstrap."""

    organization: str = "YamLabs"
    application: str = "YamImageProcessor"
    log_directory: Optional[Path] = None
    plugin_packages: Sequence[str] = field(default_factory=lambda: ["yam_processor.plugins"])
    developer_diagnostics: bool = False
    enable_console_logging: bool = True
    max_log_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 5


class AppCore:
    """Coordinates start-up of the primary services used by the application."""

    def __init__(self, config: Optional[AppConfiguration] = None) -> None:
        self.config = config or AppConfiguration()
        self.logger = logging.getLogger(__name__)
        self.logging_configurator: Optional[LoggingConfigurator] = None
        self.settings: Optional[SettingsManager] = None
        self.thread_controller: Optional[ThreadController] = None
        self.plugins: List[object] = []
        self.module_registry: ModuleRegistry = ModuleRegistry()

    def bootstrap(self) -> None:
        """Initialise all core systems."""
        self._init_logging()
        self.logger.info("Bootstrapping application core", extra={"component": "AppCore"})
        self._init_settings()
        self._init_threading()
        self._discover_plugins()

    def shutdown(self) -> None:
        """Shutdown routine for releasing resources gracefully."""
        if self.thread_controller is not None:
            self.thread_controller.shutdown()
        self.logger.info("Application core shutdown complete", extra={"component": "AppCore"})

    def _init_logging(self) -> None:
        options = LoggingOptions(
            log_directory=self.config.log_directory,
            enable_console=self.config.enable_console_logging,
            developer_diagnostics=self.config.developer_diagnostics,
            max_bytes=self.config.max_log_bytes,
            backup_count=self.config.log_backup_count,
        )
        self.logging_configurator = LoggingConfigurator(options)
        self.logging_configurator.configure()
        self.logger.debug("Logging initialised", extra={"component": "AppCore"})

    def _init_settings(self) -> None:
        self.settings = SettingsManager(self.config.organization, self.config.application)
        self.logger.debug("Settings manager initialised", extra={"component": "AppCore"})

    def _init_threading(self) -> None:
        self.thread_controller = ThreadController()
        self.logger.debug("Thread controller initialised", extra={"component": "AppCore"})

    def _discover_plugins(self) -> None:
        loader = ModuleLoader(self.config.plugin_packages)
        discovered = loader.discover()
        self.plugins = []
        for module in discovered:
            register = getattr(module, "register_module", None)
            if callable(register):
                try:
                    register(self)
                    self.plugins.append(module)
                    self.logger.info(
                        "Plugin registered", extra={"component": "AppCore", "plugin": module.__name__}
                    )
                except Exception as exc:  # pragma: no cover - plugin registration guard
                    self.logger.error(
                        "Plugin registration failed",
                        extra={"component": "AppCore", "plugin": module.__name__, "error": str(exc)},
                    )
        self.logger.debug(
            "Plugin discovery complete",
            extra={"component": "AppCore", "count": len(self.plugins)},
        )
