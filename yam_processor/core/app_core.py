"""Application core bootstrap handling logging, settings, plugins and threading."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

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
    module_paths: Sequence[Path] = field(
        default_factory=lambda: [Path(__file__).resolve().parents[2] / "modules"]
    )
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
    enable_update_checks: bool = False
    update_endpoint: Optional[str] = None
    telemetry_opt_in: bool = False
    telemetry_settings_key: str = "telemetry/opt_in"


@dataclass(frozen=True)
class UpdateMetadata:
    """Structured representation of update information returned by the endpoint."""

    version: str
    notes: Optional[str] = None
    download_url: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "UpdateMetadata":
        """Create an :class:`UpdateMetadata` instance from ``payload``.

        The endpoint is expected to return a JSON document containing at least
        a ``version`` key.  Optional fields such as ``notes`` and
        ``download_url`` are extracted when present.
        """

        if "version" not in payload or not payload["version"]:
            raise ValueError("Update metadata payload is missing a version field")

        version = str(payload["version"])
        notes = payload.get("notes") or payload.get("release_notes")
        download_url = payload.get("download_url") or payload.get("url")
        return cls(
            version=version,
            notes=str(notes) if notes is not None else None,
            download_url=str(download_url) if download_url is not None else None,
            raw={k: v for k, v in payload.items()},
        )


class UpdateDispatcher:
    """Coordinate update notifications and acknowledgement hooks."""

    def __init__(
        self,
        acknowledge_callback: Callable[[], None],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._listeners: list[Callable[[UpdateMetadata], None]] = []
        self._acknowledge_callback = acknowledge_callback
        self._pending = False
        self._logger = logger or logging.getLogger(__name__)

    def add_listener(self, callback: Callable[[UpdateMetadata], None]) -> None:
        """Register a listener to be invoked when an update is available."""

        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[UpdateMetadata], None]) -> None:
        """Unregister a previously registered listener."""

        if callback in self._listeners:
            self._listeners.remove(callback)

    def notify(self, metadata: UpdateMetadata) -> None:
        """Notify listeners of ``metadata`` and mark the dispatcher as pending."""

        self._pending = True
        for callback in list(self._listeners):
            try:
                callback(metadata)
            except Exception:  # pragma: no cover - defensive callback guard
                self._logger.exception(
                    "Update listener raised an exception",
                    extra={"component": "AppCore.UpdateDispatcher"},
                )

    def acknowledge(self) -> None:
        """Signal that the pending update notification has been acknowledged."""

        if not self._pending:
            self._logger.debug(
                "Acknowledge requested without a pending update",
                extra={"component": "AppCore.UpdateDispatcher"},
            )
            return
        self._pending = False
        try:
            self._acknowledge_callback()
        except Exception:  # pragma: no cover - defensive callback guard
            self._logger.exception(
                "Update acknowledgement callback raised",
                extra={"component": "AppCore.UpdateDispatcher"},
            )

    def has_pending_update(self) -> bool:
        """Return ``True`` when an update notification is pending acknowledgement."""

        return self._pending


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
        self.telemetry_enabled: bool = False
        self.telemetry_setting_key: str = self.config.telemetry_settings_key
        self._pending_update: Optional[UpdateMetadata] = None
        self.update_dispatcher = UpdateDispatcher(
            self._handle_update_acknowledged, logger=self.logger
        )

    def bootstrap(self) -> None:
        """Initialise all core systems."""
        self._init_session_temp_dir()
        self._configure_user_paths()
        self._init_logging()
        self.logger.info("Bootstrapping application core", extra={"component": "AppCore"})
        self._init_settings()
        self.telemetry_enabled = self._resolve_telemetry_opt_in()
        self._init_translations()
        self._init_persistence()
        self._init_threading()
        self._discover_plugins()
        if self.config.enable_update_checks:
            self.check_for_updates()
        else:
            self.logger.debug(
                "Update checks disabled by configuration",
                extra={"component": "AppCore"},
            )
        if not self.config.developer_diagnostics:
            self.logger.info(
                "Telemetry disabled (developer diagnostics disabled)",
                extra={"component": "AppCore"},
            )
        elif self.telemetry_enabled:
            self.configure_telemetry()
        else:
            self.logger.debug(
                "Telemetry disabled (opt-in not granted)",
                extra={"component": "AppCore"},
            )

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
        loader = ModuleLoader(self.config.plugin_packages, self.config.module_paths)
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

    def check_for_updates(self) -> None:
        """Invoke the update polling hook when explicitly enabled."""

        version = self._current_version()
        self.logger.info(
            "Update check requested",
            extra={"component": "AppCore", "version": version},
        )
        endpoint = self.config.update_endpoint
        if not endpoint:
            self.logger.debug(
                "No update endpoint configured; skipping check",
                extra={"component": "AppCore"},
            )
            return

        try:
            with urllib.request.urlopen(endpoint, timeout=10) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    self.logger.warning(
                        "Update endpoint returned error status",
                        extra={
                            "component": "AppCore",
                            "status": status,
                            "endpoint": endpoint,
                        },
                    )
                    return
                payload_bytes = response.read()
        except urllib.error.URLError as exc:
            self.logger.warning(
                "Failed to contact update endpoint",
                extra={"component": "AppCore", "endpoint": endpoint, "error": str(exc)},
            )
            return

        try:
            data = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.logger.warning(
                "Update endpoint returned invalid JSON",
                extra={"component": "AppCore", "endpoint": endpoint, "error": str(exc)},
            )
            return

        if not isinstance(data, dict):
            self.logger.warning(
                "Update endpoint returned unexpected payload",
                extra={"component": "AppCore", "endpoint": endpoint, "payload_type": type(data).__name__},
            )
            return

        try:
            metadata = UpdateMetadata.from_payload(data)
        except ValueError as exc:
            self.logger.warning(
                "Update payload missing required fields",
                extra={"component": "AppCore", "endpoint": endpoint, "error": str(exc)},
            )
            return

        if metadata.version == version:
            self.logger.debug(
                "Application already at latest version",
                extra={"component": "AppCore", "version": version},
            )
            return

        self.logger.info(
            "Update available",
            extra={
                "component": "AppCore",
                "current_version": version,
                "available_version": metadata.version,
            },
        )
        self._handle_update_available(metadata)

    def configure_telemetry(self) -> None:
        """Enable telemetry emission when the user has opted in."""

        if not self.config.developer_diagnostics:
            if self.settings is not None:
                self.settings.set(self.telemetry_setting_key, False)
            self.telemetry_enabled = False
            self.logger.debug(
                "Telemetry configuration skipped (developer diagnostics disabled)",
                extra={"component": "AppCore"},
            )
            return

        if self.settings is None:
            self.logger.debug(
                "Telemetry configuration skipped (no settings manager)",
                extra={"component": "AppCore"},
            )
            return

        key = self.telemetry_setting_key
        self.settings.set(key, True)
        self.telemetry_enabled = True
        version = self._current_version()
        self.logger.info(
            "Telemetry opt-in active",
            extra={"component": "AppCore", "version": version, "settings_key": key},
        )

    def _resolve_telemetry_opt_in(self) -> bool:
        if not self.config.developer_diagnostics:
            if self.settings is not None:
                self.settings.set(self.telemetry_setting_key, False)
            return False

        if self.settings is None:
            return bool(self.config.telemetry_opt_in)

        key = self.telemetry_setting_key
        sentinel = object()
        stored_value = self.settings.get(key, sentinel)
        if stored_value is sentinel:
            enabled = bool(self.config.telemetry_opt_in)
            self.settings.set(key, enabled)
            return enabled

        try:
            enabled = self._coerce_bool(stored_value)
        except ValueError:
            enabled = bool(self.config.telemetry_opt_in)
            self.settings.set(key, enabled)
            self.logger.warning(
                "Invalid telemetry opt-in setting encountered; reverting to default",
                extra={"component": "AppCore", "settings_key": key},
            )
        else:
            if enabled not in (True, False):  # pragma: no cover - defensive branch
                enabled = bool(enabled)
        self.settings.set(key, enabled)
        return enabled

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in {"1", "true", "yes", "on"}:
                return True
            if normalised in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Cannot interpret string as boolean: {value!r}")
        return bool(value)

    @staticmethod
    def _current_version() -> str:
        try:
            from yam_processor import get_version
        except Exception:  # pragma: no cover - circular import safety
            return "unknown"
        try:
            return get_version()
        except Exception:  # pragma: no cover - fallback handling
            return "unknown"

    # ------------------------------------------------------------------
    # Update handling helpers
    # ------------------------------------------------------------------
    def _handle_update_available(self, metadata: UpdateMetadata) -> None:
        self._pending_update = metadata
        if self.thread_controller is not None:
            self.thread_controller.pause()
        self.update_dispatcher.notify(metadata)

    def _handle_update_acknowledged(self) -> None:
        if self._pending_update is None:
            self.logger.debug(
                "Update acknowledgement received without pending metadata",
                extra={"component": "AppCore"},
            )
            return
        acknowledged_version = self._pending_update.version
        self._pending_update = None
        if self.thread_controller is not None:
            self.thread_controller.resume()
        self.logger.info(
            "Update acknowledgement received",
            extra={"component": "AppCore", "acknowledged_version": acknowledged_version},
        )

    def acknowledge_update(self) -> None:
        """Public helper to acknowledge a pending update notification."""

        self.update_dispatcher.acknowledge()
