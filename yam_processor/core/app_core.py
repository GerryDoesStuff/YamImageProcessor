"""Application core bootstrap handling logging, settings, plugins and threading."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from PyQt5 import QtCore  # type: ignore

from yam_processor.data import configure_allowed_roots

from .logging_config import LoggingConfigurator, LoggingOptions
from .module_loader import ModuleLoader, ModuleRegistry
from .persistence import AutosaveManager
from .recovery import RecoveryManager
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
    session_temp_parent: Optional[Path] = None
    allowed_user_roots: Sequence[Path] = field(default_factory=lambda: [Path.home(), Path.cwd()])
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
        self.recovery_manager: Optional[RecoveryManager] = None
        self.plugins: List[object] = []
        self.module_registry: ModuleRegistry = ModuleRegistry()
        self.translators: list[QtCore.QTranslator] = []
        self.session_temp_dir: Optional[Path] = None

    def bootstrap(self) -> None:
        """Initialise all core systems."""
        self._init_session_temp_dir()
        self._configure_user_paths()
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
        if self.recovery_manager is not None:
            self.recovery_manager.cleanup_crash_markers()
        self._remove_translations()
        self._teardown_session_temp_dir()
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

    def _init_session_temp_dir(self) -> None:
        if self.session_temp_dir is not None:
            return
        prefix = f"{self.config.application.lower()}_"
        parent = self.config.session_temp_parent
        dir_arg = os.fspath(parent) if parent is not None else None
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=dir_arg))
        cache_dir = path / "pipeline_cache"
        recovery_dir = path / "recovery"
        cache_dir.mkdir(parents=True, exist_ok=True)
        recovery_dir.mkdir(parents=True, exist_ok=True)
        self.session_temp_dir = path
        try:
            from yam_processor.processing.pipeline_manager import PipelineManager

            PipelineManager.set_default_cache_directory(cache_dir)
            PipelineManager.set_default_recovery_root(recovery_dir)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug(
                "Failed to configure pipeline cache directories", extra={"error": str(exc)}
            )
        self.logger.debug(
            "Session temporary directory initialised",
            extra={"component": "AppCore", "temp_dir": str(path)},
        )

    def _configure_user_paths(self) -> None:
        roots = [Path(root).expanduser() for root in self.config.allowed_user_roots]
        try:
            configure_allowed_roots(roots)
        except ValueError as exc:
            self.logger.warning(
                "Invalid allowed roots configuration", extra={"error": str(exc)}
            )
            configure_allowed_roots([Path.cwd()])

    def _teardown_session_temp_dir(self) -> None:
        if self.session_temp_dir is None:
            return
        try:
            shutil.rmtree(self.session_temp_dir, ignore_errors=True)
        finally:
            self.logger.debug(
                "Session temporary directory removed",
                extra={"component": "AppCore", "temp_dir": str(self.session_temp_dir)},
            )
            self.session_temp_dir = None

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
        recovery_root = self.session_temp_dir / "recovery" if self.session_temp_dir else None
        recovery_logger = logging.getLogger(f"{__name__}.Recovery")
        self.recovery_manager = RecoveryManager(
            autosave_dir,
            crash_marker_root=recovery_root,
            logger=recovery_logger,
        )
        self.recovery_manager.inspect_startup()
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
