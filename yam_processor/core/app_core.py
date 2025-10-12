"""Application core bootstrap handling logging, settings, plugins and threading."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from PyQt5 import QtCore  # type: ignore

from .logging_config import LoggingConfigurator, LoggingOptions
from .module_loader import ModuleLoader, ModuleRegistry
from .persistence import AutosaveManager
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
    autosave_directory: Optional[Path] = None
    autosave_interval_seconds: float = 120.0
    autosave_backup_retention: int = 5
    translation_directories: Sequence[Path] = field(
        default_factory=lambda: [Path(__file__).resolve().parents[1] / "i18n"]
    )
    translation_locales: Sequence[str] = ()
    translation_file_prefix: str = "yam_processor"


class AppCore:
    """Coordinates start-up of the primary services used by the application."""

    def __init__(self, config: Optional[AppConfiguration] = None) -> None:
        self.config = config or AppConfiguration()
        self.logger = logging.getLogger(__name__)
        self.logging_configurator: Optional[LoggingConfigurator] = None
        self.settings: Optional[SettingsManager] = None
        self.thread_controller: Optional[ThreadController] = None
        self.autosave_manager: Optional[AutosaveManager] = None
        self.plugins: List[object] = []
        self.module_registry: ModuleRegistry = ModuleRegistry()
        self.translators: list[QtCore.QTranslator] = []

    def bootstrap(self) -> None:
        """Initialise all core systems."""
        self._init_logging()
        self.logger.info("Bootstrapping application core", extra={"component": "AppCore"})
        self._init_settings()
        self._init_translations()
        self._init_persistence()
        self._init_threading()
        self._discover_plugins()

    def shutdown(self) -> None:
        """Shutdown routine for releasing resources gracefully."""
        if self.thread_controller is not None:
            self.thread_controller.shutdown()
        if self.autosave_manager is not None:
            self.autosave_manager.shutdown()
        self._remove_translations()
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

    def _init_translations(self) -> None:
        app = QtCore.QCoreApplication.instance()
        if app is None:
            self.logger.debug(
                "Translation initialisation skipped (no QCoreApplication)",
                extra={"component": "AppCore"},
            )
            return

        self._remove_translations()

        locale_candidates = list(self.config.translation_locales)
        if not locale_candidates:
            system_locale = QtCore.QLocale.system()
            locale_candidates.extend(system_locale.uiLanguages())
            locale_candidates.append(system_locale.name())

        seen: set[str] = set()
        normalised_locales: list[str] = []
        for locale in locale_candidates:
            if not locale:
                continue
            key = locale.replace("-", "_")
            if key not in seen:
                seen.add(key)
                normalised_locales.append(key)

        directories = [Path(path) for path in self.config.translation_directories]
        loaded_files: list[str] = []
        loaded_paths: set[Path] = set()
        for locale in normalised_locales:
            candidates = self._translation_file_candidates(locale, directories)
            for candidate in candidates:
                if candidate in loaded_paths:
                    continue
                translator = QtCore.QTranslator(app)
                if translator.load(str(candidate)):
                    app.installTranslator(translator)
                    self.translators.append(translator)
                    loaded_files.append(candidate.name)
                    loaded_paths.add(candidate)
                else:
                    self.logger.debug(
                        "Failed to load translation file",
                        extra={
                            "component": "AppCore",
                            "file": str(candidate),
                            "locale": locale,
                        },
                    )
        if loaded_files:
            self.logger.info(
                "Loaded translations",
                extra={"component": "AppCore", "files": ", ".join(loaded_files)},
            )
        else:
            self.logger.debug(
                "No translation files loaded",
                extra={"component": "AppCore"},
            )

    def _translation_file_candidates(
        self, locale: str, directories: Sequence[Path]
    ) -> list[Path]:
        prefix = self.config.translation_file_prefix
        patterns: list[str] = []
        if "_" in locale:
            language, territory = locale.split("_", 1)
            patterns.append(f"{language}_{territory}")
            patterns.append(language)
        else:
            patterns.append(locale)
        patterns = [p for p in patterns if p]
        candidates: list[Path] = []
        for directory in directories:
            if not directory.exists():
                continue
            for pattern in patterns:
                file_name = f"{prefix}_{pattern}.qm"
                path = directory / file_name
                if path.exists() and path not in candidates:
                    candidates.append(path)
        return candidates

    def _remove_translations(self) -> None:
        app = QtCore.QCoreApplication.instance()
        if app is None:
            self.translators.clear()
            return
        while self.translators:
            translator = self.translators.pop()
            app.removeTranslator(translator)

    def _init_persistence(self) -> None:
        autosave_dir = (
            self.config.autosave_directory
            or Path.home() / f".{self.config.application.lower()}" / "autosave"
        )
        autosave_logger = logging.getLogger(f"{__name__}.Autosave")
        self.autosave_manager = AutosaveManager(
            autosave_dir,
            interval_seconds=self.config.autosave_interval_seconds,
            backup_retention=self.config.autosave_backup_retention,
            logger=autosave_logger,
        )
        self.logger.debug(
            "Autosave manager initialised",
            extra={"component": "AppCore", "autosave_dir": str(autosave_dir)},
        )

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
