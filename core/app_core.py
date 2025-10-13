"""Lightweight application bootstrap exposing shared services."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from yam_processor.core.module_loader import ModuleLoader, ModuleRegistry
from yam_processor.core.settings_manager import SettingsManager
from yam_processor.core.threading import ThreadController


@dataclass
class AppConfiguration:
    """Configuration values used when constructing :class:`AppCore`."""

    organization: str = "MicroscopicApp"
    application: str = "ImageProcessor"
    log_level: int = logging.INFO
    plugin_packages: Sequence[str] = field(default_factory=lambda: ["plugins"])
    module_paths: Sequence[Path | str] = field(default_factory=list)
    max_workers: Optional[int] = None


class AppCore:
    """Provide access to process-wide services such as settings and threading."""

    def __init__(self, config: Optional[AppConfiguration] = None) -> None:
        self.config = config or AppConfiguration()
        self.logger = logging.getLogger(__name__)
        self.settings_manager: Optional[SettingsManager] = None
        self.thread_controller: Optional[ThreadController] = None
        self.module_registry: ModuleRegistry = ModuleRegistry()
        self.plugins: List[object] = []
        self._bootstrapped = False

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

    def ensure_bootstrapped(self) -> None:
        """Ensure :meth:`bootstrap` has been executed."""

        if not self._bootstrapped:
            self.bootstrap()

    # ------------------------------------------------------------------
    # Internal helpers
    def _configure_logging(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=self.config.log_level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
        else:
            logging.getLogger().setLevel(self.config.log_level)

    def _init_settings(self) -> None:
        self.settings_manager = SettingsManager(
            self.config.organization,
            self.config.application,
        )
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

    def _discover_plugins(self) -> None:
        loader = ModuleLoader(
            self.config.plugin_packages,
            [Path(path) for path in self.config.module_paths],
        )
        self.plugins = []
        for module in loader.discover():
            register = getattr(module, "register_module", None)
            if callable(register):
                try:
                    register(self)
                    self.logger.debug(
                        "Plugin registered",
                        extra={
                            "component": "AppCore",
                            "module": module.__name__,
                        },
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
                    continue
            self.plugins.append(module)
        self.logger.debug(
            "Plugin discovery complete",
            extra={"component": "AppCore", "count": len(self.plugins)},
        )


__all__ = ["AppConfiguration", "AppCore"]
