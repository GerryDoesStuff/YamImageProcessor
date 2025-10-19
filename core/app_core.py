"""Lightweight application bootstrap exposing shared services."""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import pkgutil
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from plugins.module_base import ModuleBase, ModuleMetadata, ModuleStage

from .thread_controller import ThreadController

from .io_manager import IOManager
from .path_sanitizer import PathValidationError, allowed_roots as sanitizer_allowed_roots
from .path_sanitizer import configure_allowed_roots
from .persistence import AutosaveManager
from .recovery import RecoveryManager
from .settings import SettingsManager
from .signing import (
    InvalidSignatureError,
    MissingSignatureError,
    ModuleSignatureVerifier,
    SignatureVerificationError,
    TrustStoreError,
    signature_path_for,
)
from .i18n import default_translation_directories

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
    telemetry_opt_in: bool = False
    telemetry_settings_key: str = "telemetry/enabled"
    log_directory: Path = Path("logs")
    log_filename: str = "application.log"
    plugin_packages: Sequence[str] = field(default_factory=lambda: ["modules"])
    module_paths: Sequence[Path | str] = field(default_factory=list)
    plugin_trust_store_paths: Sequence[Path | str] = field(default_factory=list)
    plugin_signature_extension: str = ".sig"
    max_workers: Optional[int] = None
    autosave_directory: Optional[Path] = None
    autosave_interval_seconds: float = 120.0
    autosave_backup_retention: int = 5
    autosave_enabled_default: bool = True
    allowed_roots: Sequence[Path | str] = field(default_factory=lambda: [Path.cwd()])
    translation_directories: Sequence[Path | str] = field(
        default_factory=default_translation_directories
    )
    translation_locales: Sequence[str] = field(default_factory=tuple)
    translation_prefix: str = "yam_processor"
    session_temp_enabled: bool = True
    session_temp_parent: Optional[Path | str] = None
    session_temp_cleanup_on_shutdown: bool = True
    enable_update_checks: bool = False
    update_endpoint: Optional[str] = None


@dataclass(frozen=True)
class UpdateMetadata:
    """Structured representation of application update information."""

    version: str
    notes: Optional[str] = None
    release_notes_url: Optional[str] = None
    download_url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "UpdateMetadata":
        """Create :class:`UpdateMetadata` from a JSON payload."""

        if "version" not in payload or not payload["version"]:
            raise ValueError("Update payload missing required 'version' field")

        version = str(payload["version"])
        notes = payload.get("notes")
        release_notes = payload.get("release_notes")
        release_notes_url = (
            payload.get("release_notes_url")
            or payload.get("notes_url")
            or payload.get("changelog_url")
        )

        if isinstance(notes, dict):
            release_notes_url = release_notes_url or notes.get("url")
            notes = notes.get("text")

        if notes is None and isinstance(release_notes, dict):
            notes = release_notes.get("text")
            release_notes_url = release_notes_url or release_notes.get("url")
        elif notes is None and release_notes is not None:
            notes = release_notes

        download_url = payload.get("download_url") or payload.get("url")

        return cls(
            version=version,
            notes=str(notes) if notes is not None else None,
            release_notes_url=(
                str(release_notes_url) if release_notes_url is not None else None
            ),
            download_url=str(download_url) if download_url is not None else None,
            raw=dict(payload),
        )


class UpdateDispatcher:
    """Coordinate update notifications and acknowledgement callbacks."""

    def __init__(
        self,
        acknowledge_callback: Callable[[], None],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._listeners: List[Callable[[UpdateMetadata], None]] = []
        self._acknowledge_callback = acknowledge_callback
        self._pending = False
        self._logger = logger or logging.getLogger(__name__)

    def add_listener(self, callback: Callable[[UpdateMetadata], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[UpdateMetadata], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def notify(self, metadata: UpdateMetadata) -> None:
        self._pending = True
        for callback in list(self._listeners):
            try:
                callback(metadata)
            except Exception:  # pragma: no cover - defensive guard for UI callbacks
                self._logger.exception(
                    "Update listener raised an exception", extra={"component": "AppCore"}
                )

    def acknowledge(self) -> None:
        if not self._pending:
            return
        self._pending = False
        try:
            self._acknowledge_callback()
        except Exception:  # pragma: no cover - defensive guard for callbacks
            self._logger.exception(
                "Update acknowledgement callback failed", extra={"component": "AppCore"}
            )

    def has_pending_update(self) -> bool:
        return self._pending


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
        self.recovery_manager: Optional[RecoveryManager] = None
        self._module_catalog: Dict[ModuleStage, Dict[str, ModuleBase]] = {
            stage: {} for stage in ModuleStage
        }
        self._module_enabled: Dict[ModuleStage, Dict[str, bool]] = {
            stage: {} for stage in ModuleStage
        }
        self.plugins: List[object] = []
        self._log_handler: Optional[logging.Handler] = None
        self._bootstrapped = False
        self._preprocessing_manager: Optional[PipelineManager] = None
        self._preprocessing_templates: Dict[str, PipelineStep] = {}
        self._pipeline_manager: Optional[PipelineManager] = None
        self._pipeline_stage_templates: Dict[
            ModuleStage, Dict[str, PipelineStep]
        ] = {entry_stage: {} for entry_stage in ModuleStage}
        self._pipeline_stage_ranges: Dict[ModuleStage, Tuple[int, int]] = {}
        self._pipeline_cache: Optional[PipelineCache] = None
        self.session_temp_root: Optional[Path] = None
        self.session_pipeline_cache_dir: Optional[Path] = None
        self.session_recovery_dir: Optional[Path] = None
        self.autosave_workspace: Optional[Path] = None
        self.telemetry_setting_key: str = self.config.telemetry_settings_key
        self._telemetry_opt_in: bool = self._coerce_bool(self.config.telemetry_opt_in)
        self.telemetry_enabled: bool = False
        self._pending_update: Optional[UpdateMetadata] = None
        self.update_dispatcher = UpdateDispatcher(
            self._handle_update_acknowledged, logger=self.logger
        )
        self._signature_extension = self.config.plugin_signature_extension
        self._signature_verifier: Optional[ModuleSignatureVerifier] = None
        if self.config.plugin_trust_store_paths:
            try:
                self._signature_verifier = ModuleSignatureVerifier(
                    self.config.plugin_trust_store_paths
                )
            except TrustStoreError as exc:
                self.logger.warning(
                    "Failed to initialise plugin signature verifier",
                    extra={
                        "component": "AppCore",
                        "error": str(exc),
                    },
                )

    # ------------------------------------------------------------------
    # Lifecycle management
    def bootstrap(self) -> None:
        """Initialise logging, settings, threading and plugin discovery."""

        if self._bootstrapped:
            return

        self._prepare_session_temp_root()
        self._refresh_allowed_roots()
        self._configure_logging()
        self._init_settings()
        self._init_threading()
        self._discover_plugins()
        if self.config.enable_update_checks:
            self.check_for_updates()
        else:
            self.logger.debug(
                "Update checks disabled by configuration",
                extra={"component": "AppCore"},
            )
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

        if self.recovery_manager is not None:
            self.recovery_manager.cleanup_crash_markers()
            self.recovery_manager = None

        self._pipeline_cache = None
        self._preprocessing_manager = None
        self._preprocessing_templates = {}
        self._pipeline_manager = None
        self._pipeline_stage_templates = {
            entry_stage: {} for entry_stage in ModuleStage
        }
        self._pipeline_stage_ranges = {}
        self._teardown_session_temp_root()
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
    def allowed_roots(self) -> tuple[Path, ...]:
        """Expose the currently configured sandbox roots."""

        return sanitizer_allowed_roots()

    def refresh_allowed_roots(self) -> None:
        """Recalculate the configured allowed roots."""

        self._refresh_allowed_roots()

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

    @property
    def recovery(self) -> RecoveryManager:
        """Return the crash recovery manager."""

        if self.recovery_manager is None:
            raise RuntimeError("Recovery manager not initialised. Call bootstrap() first.")
        return self.recovery_manager

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
            self._pipeline_cache = PipelineCache(
                self.settings_manager,
                cache_directory=self.session_pipeline_cache_dir,
            )
        return self._pipeline_cache

    def get_pipeline_manager(self) -> PipelineManager:
        """Return the unified pipeline manager spanning all module stages."""

        if self._pipeline_manager is None:
            manager, templates, ranges = self._build_pipeline_manager()
            self._pipeline_manager = manager
            self._pipeline_stage_templates = templates
            self._pipeline_stage_ranges = ranges
            self._preprocessing_templates = templates.get(
                ModuleStage.PREPROCESSING, {}
            )
            # Invalidate stage specific clones so they can be rebuilt from the
            # refreshed templates on demand.
            self._preprocessing_manager = None
        return self._pipeline_manager

    def pipeline_stage_bounds(self, stage: ModuleStage) -> Tuple[int, int]:
        """Return the start/end indices for ``stage`` within the unified pipeline."""

        self.get_pipeline_manager()
        return self._pipeline_stage_ranges.get(stage, (0, 0))

    def pipeline_stage_templates(self, stage: ModuleStage) -> Dict[str, PipelineStep]:
        """Return the cached step templates for ``stage``."""

        self.get_pipeline_manager()
        return self._pipeline_stage_templates.get(stage, {})

    def get_preprocessing_pipeline_manager(self) -> PipelineManager:
        if self._preprocessing_manager is None:
            self.get_pipeline_manager()
            templates = self._pipeline_stage_templates.get(ModuleStage.PREPROCESSING, {})
            self._preprocessing_templates = {
                name: step.clone() for name, step in templates.items()
            }
            self._preprocessing_manager = PipelineManager(
                (step.clone() for step in templates.values()),
                cache_dir=self.session_pipeline_cache_dir,
                recovery_root=self.session_recovery_dir,
            )
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
    def _build_pipeline_manager(
        self,
    ) -> Tuple[PipelineManager, Dict[ModuleStage, Dict[str, PipelineStep]], Dict[ModuleStage, Tuple[int, int]]]:
        steps: List[PipelineStep] = []
        templates: Dict[ModuleStage, Dict[str, PipelineStep]] = {}
        ranges: Dict[ModuleStage, Tuple[int, int]] = {}

        for stage in ModuleStage:
            stage_modules = sorted(
                self.iter_enabled_modules(stage), key=self._module_sort_key
            )
            stage_steps = [
                self._coerce_stage_step(module, stage) for module in stage_modules
            ]
            templates[stage] = {step.name: step.clone() for step in stage_steps}
            start = len(steps)
            steps.extend(step.clone() for step in stage_steps)
            ranges[stage] = (start, len(steps))

        manager = PipelineManager(
            steps,
            cache_dir=self.session_pipeline_cache_dir,
            recovery_root=self.session_recovery_dir,
        )
        return manager, templates, ranges

    def _coerce_stage_step(self, module: ModuleBase, stage: ModuleStage) -> PipelineStep:
        step = module.create_pipeline_step()
        step.stage = stage
        return step

    def _module_sort_key(self, module: ModuleBase) -> Tuple[Any, ...]:
        metadata = module.metadata
        menu_path = tuple(str(part).lower() for part in metadata.menu_path)
        return (*menu_path, metadata.title.lower(), metadata.identifier.lower())

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
            self.telemetry_setting_key: self._telemetry_opt_in,
        }
        if self.config.autosave_directory is not None:
            defaults["autosave/workspace"] = str(self.config.autosave_directory)

        self.settings_manager = SettingsManager(
            self.config.organization,
            self.config.application,
            defaults=defaults,
        )
        self._pipeline_cache = PipelineCache(
            self.settings_manager,
            cache_directory=self.session_pipeline_cache_dir,
        )
        stored_diagnostics = self.settings_manager.get(
            "diagnostics/enabled", self.config.diagnostics_enabled
        )
        self.set_diagnostics_enabled(
            self._coerce_bool(stored_diagnostics), persist=False
        )
        stored_opt_in = self.settings_manager.get(
            self.telemetry_setting_key, self._telemetry_opt_in
        )
        requested_opt_in = self._coerce_bool(stored_opt_in)
        self._telemetry_opt_in = requested_opt_in
        self.telemetry_enabled = self._resolve_telemetry_opt_in(requested_opt_in)
        self._io_manager = IOManager(self.settings_manager)
        self._refresh_allowed_roots()
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
        workspace = Path(workspace)
        self.autosave_workspace = workspace

        recovery_logger = logging.getLogger(f"{__name__}.Recovery")
        self.recovery_manager = RecoveryManager(
            workspace,
            recovery_root=self.session_recovery_dir,
            logger=recovery_logger,
        )
        self.recovery_manager.inspect_startup()
        autosave_logger = logging.getLogger(f"{__name__}.Autosave")
        self.autosave_manager = AutosaveManager(
            self.settings_manager,
            self._io_manager,
            autosave_directory=workspace,
            interval_seconds=self.settings_manager.autosave_interval(),
            logger=autosave_logger,
            recovery_manager=self.recovery_manager,
        )
        self._refresh_allowed_roots()
        self.logger.debug(
            "Autosave manager initialised",
            extra={"component": "AppCore", "autosave_dir": str(workspace)},
        )
        self.logger.debug(
            "Recovery manager initialised",
            extra={
                "component": "AppCore",
                "autosave_dir": str(workspace),
                "recovery_root": str(self.session_recovery_dir) if self.session_recovery_dir else None,
            },
        )

    def _read_module_bytes(self, spec: Any, module_path: Path) -> Optional[bytes]:
        loader = getattr(spec, "loader", None)
        origin = getattr(spec, "origin", None)
        if loader and origin and hasattr(loader, "get_data"):
            try:
                return loader.get_data(origin)  # type: ignore[call-arg]
            except OSError:
                pass

        try:
            return module_path.read_bytes()
        except OSError:
            return None

    def _should_load_module(self, module_name: str) -> bool:
        if self._signature_verifier is None:
            return True

        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ModuleNotFoundError) as exc:
            self.logger.warning(
                "Unable to resolve module for signature verification",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "error": str(exc),
                },
            )
            return False

        if spec is None or getattr(spec, "origin", None) is None:
            self.logger.warning(
                "Module has no origin for signature verification",
                extra={"component": "AppCore", "plugin_module": module_name},
            )
            return False

        module_path = Path(spec.origin)
        module_bytes = self._read_module_bytes(spec, module_path)
        if module_bytes is None:
            self.logger.warning(
                "Failed to read module bytes for signature verification",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "path": str(module_path),
                },
            )
            return False

        signature_path = signature_path_for(module_path, self._signature_extension)
        try:
            signature_bytes = signature_path.read_bytes()
        except FileNotFoundError:
            self.logger.warning(
                "Missing signature for module",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "signature_path": str(signature_path),
                },
            )
            return False
        except OSError as exc:
            self.logger.warning(
                "Unable to read module signature",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "signature_path": str(signature_path),
                    "error": str(exc),
                },
            )
            return False

        try:
            self._signature_verifier.verify(module_bytes, signature_bytes)
        except MissingSignatureError:
            self.logger.warning(
                "Empty signature provided for module",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "signature_path": str(signature_path),
                },
            )
            return False
        except InvalidSignatureError:
            self.logger.warning(
                "Rejected module due to invalid signature",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "signature_path": str(signature_path),
                },
            )
            return False
        except SignatureVerificationError as exc:
            self.logger.warning(
                "Failed to verify module signature",
                extra={
                    "component": "AppCore",
                    "plugin_module": module_name,
                    "signature_path": str(signature_path),
                    "error": str(exc),
                },
            )
            return False

        return True

    def _discover_plugins(self) -> None:
        discovered: Dict[str, ModuleType] = {}
        for package_name in self.config.plugin_packages:
            if not self._should_load_module(package_name):
                continue
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
                if not self._should_load_module(name):
                    continue
                try:
                    module = importlib.import_module(name)
                except Exception as exc:  # pragma: no cover - plugin import guard
                    self.logger.warning(
                        "Failed to load plugin module",
                        extra={
                            "component": "AppCore",
                            "plugin_module": name,
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
                extra={"component": "AppCore", "plugin_module": module.__name__},
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Plugin registration failed",
                extra={
                    "component": "AppCore",
                    "plugin_module": module.__name__,
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
        enabled_map = self._module_enabled.setdefault(metadata.stage, {})
        if metadata.identifier not in enabled_map:
            enabled_map[metadata.identifier] = self._resolve_module_enabled_default(
                metadata
            )
        self.logger.info(
            "Registered module",
            extra={
                "component": "AppCore",
                "identifier": metadata.identifier,
                "stage": metadata.stage.value,
            },
        )
        self._pipeline_manager = None
        self._pipeline_stage_templates = {
            entry_stage: {} for entry_stage in ModuleStage
        }
        self._pipeline_stage_ranges = {}
        self._preprocessing_manager = None
        self._preprocessing_templates = {}

    def _module_settings_key(self, stage: ModuleStage, identifier: str) -> str:
        return f"modules/{stage.value}/{identifier}/enabled"

    def _resolve_module_enabled_default(self, metadata: ModuleMetadata) -> bool:
        if self.settings_manager is None:
            return True
        key = self._module_settings_key(metadata.stage, metadata.identifier)
        if self.settings_manager.contains(key):
            return self.settings_manager.get_bool(key, True)
        return True

    def iter_modules(self, stage: ModuleStage | None = None) -> Iterator[ModuleBase]:
        """Yield registered modules, optionally filtered by ``stage``."""

        if stage is None:
            for modules in self._module_catalog.values():
                yield from modules.values()
            return
        yield from self._module_catalog.get(stage, {}).values()

    def iter_enabled_modules(self, stage: ModuleStage | None = None) -> Iterator[ModuleBase]:
        """Yield only the modules that are currently enabled."""

        if stage is None:
            for entry_stage in ModuleStage:
                yield from self.iter_enabled_modules(entry_stage)
            return

        modules = self._module_catalog.get(stage, {})
        enabled_map = self._module_enabled.get(stage, {})
        for identifier, module in modules.items():
            if enabled_map.get(identifier, True):
                yield module

    def get_modules(self, stage: ModuleStage) -> Tuple[ModuleBase, ...]:
        """Return the registered modules for ``stage``."""

        return tuple(self.iter_enabled_modules(stage))

    def enabled_modules(self, stage: ModuleStage | None = None) -> Tuple[ModuleBase, ...]:
        """Return the currently enabled modules as a tuple."""

        if stage is None:
            return tuple(self.iter_enabled_modules())
        return tuple(self.iter_enabled_modules(stage))

    def module_enabled(self, stage: ModuleStage, identifier: str) -> bool:
        """Return whether ``identifier`` for ``stage`` is enabled."""

        return self._module_enabled.get(stage, {}).get(identifier, True)

    def set_module_enabled(
        self,
        stage: ModuleStage,
        identifier: str,
        enabled: bool,
        *,
        persist: bool = True,
    ) -> None:
        """Persist the enabled flag for ``identifier`` within ``stage``."""

        enabled = bool(enabled)
        stage_modules = self._module_catalog.get(stage, {})
        if identifier not in stage_modules:
            return

        enabled_map = self._module_enabled.setdefault(stage, {})
        previous = enabled_map.get(identifier)
        enabled_map[identifier] = enabled

        if persist and self.settings_manager is not None:
            self.settings_manager.set(
                self._module_settings_key(stage, identifier), enabled
            )

        if previous == enabled:
            return

        self._pipeline_manager = None
        self._pipeline_stage_templates = {
            entry_stage: {} for entry_stage in ModuleStage
        }
        self._pipeline_stage_ranges = {}
        self._preprocessing_manager = None
        self._preprocessing_templates = {}

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
        self.configure_telemetry(self._telemetry_opt_in, persist=False)

    def configure_telemetry(self, opt_in: bool, *, persist: bool = True) -> None:
        requested = self._coerce_bool(opt_in)
        previous_state = self.telemetry_enabled
        self._telemetry_opt_in = requested
        telemetry_state = self._resolve_telemetry_opt_in(requested)
        diagnostics_active = self.config.diagnostics_enabled
        self.telemetry_enabled = telemetry_state

        if telemetry_state and not previous_state:
            self.logger.info(
                "Telemetry enabled",
                extra={"component": "AppCore", "telemetry_enabled": True},
            )
        elif previous_state and not telemetry_state:
            self.logger.info(
                "Telemetry disabled",
                extra={
                    "component": "AppCore",
                    "telemetry_enabled": False,
                    "diagnostics_enabled": diagnostics_active,
                    "opt_in": requested,
                },
            )
        elif requested and not diagnostics_active:
            self.logger.debug(
                "Telemetry opt-in ignored (diagnostics disabled)",
                extra={"component": "AppCore"},
            )

        if persist and self.settings_manager is not None:
            self.settings_manager.set(self.telemetry_setting_key, requested)

    @property
    def telemetry_opt_in(self) -> bool:
        return self._telemetry_opt_in

    def _resolve_telemetry_opt_in(self, requested: Optional[bool] = None) -> bool:
        if requested is None:
            requested_value = self._telemetry_opt_in
            if self.settings_manager is not None:
                stored = self.settings_manager.get(
                    self.telemetry_setting_key, requested_value
                )
                requested_value = self._coerce_bool(stored)
        else:
            requested_value = self._coerce_bool(requested)

        self._telemetry_opt_in = requested_value

        if requested_value and not self.config.diagnostics_enabled:
            self.logger.debug(
                "Telemetry opt-in suppressed (diagnostics disabled)",
                extra={"component": "AppCore"},
            )
            return False
        return requested_value and self.config.diagnostics_enabled

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

    def _prepare_session_temp_root(self) -> None:
        if not self.config.session_temp_enabled:
            self.logger.debug(
                "Session temporary directories disabled by configuration",
                extra={"component": "AppCore"},
            )
            return

        if self.session_temp_root is not None:
            return

        prefix = f"{self.config.application.lower()}_session_"
        parent = self.config.session_temp_parent
        dir_arg = str(parent) if parent is not None else None

        try:
            root = Path(tempfile.mkdtemp(prefix=prefix, dir=dir_arg))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Failed to create session temporary directory",
                extra={"component": "AppCore", "error": str(exc)},
            )
            return

        pipeline_cache_dir = root / "pipeline_cache"
        recovery_dir = root / "recovery"

        try:
            pipeline_cache_dir.mkdir(parents=True, exist_ok=True)
            recovery_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Failed to initialise session temporary subdirectories",
                extra={"component": "AppCore", "temp_dir": str(root), "error": str(exc)},
            )
            try:
                shutil.rmtree(root, ignore_errors=True)
            finally:
                return

        self.session_temp_root = root
        self.session_pipeline_cache_dir = pipeline_cache_dir
        self.session_recovery_dir = recovery_dir

        self._refresh_allowed_roots()

        PipelineCache.set_default_cache_directory(pipeline_cache_dir)
        PipelineManager.set_default_cache_directory(pipeline_cache_dir)
        PipelineManager.set_default_recovery_root(recovery_dir)

        self.logger.debug(
            "Session temporary directories initialised",
            extra={
                "component": "AppCore",
                "temp_root": str(root),
                "pipeline_cache_dir": str(pipeline_cache_dir),
                "recovery_dir": str(recovery_dir),
            },
        )

    def _teardown_session_temp_root(self) -> None:
        root = self.session_temp_root
        if root is None:
            return

        PipelineCache.set_default_cache_directory(None)
        PipelineManager.set_default_cache_directory(None)
        PipelineManager.set_default_recovery_root(None)

        should_cleanup = self.config.session_temp_cleanup_on_shutdown
        if should_cleanup:
            try:
                shutil.rmtree(root, ignore_errors=False)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Failed to remove session temporary directory",
                    extra={
                        "component": "AppCore",
                        "temp_root": str(root),
                        "error": str(exc),
                    },
                )
            else:
                self.logger.debug(
                    "Session temporary directory removed",
                    extra={"component": "AppCore", "temp_root": str(root)},
                )
        else:
            self.logger.debug(
                "Preserving session temporary directory on shutdown",
                extra={"component": "AppCore", "temp_root": str(root)},
            )

        self.session_temp_root = None
        self.session_pipeline_cache_dir = None
        self.session_recovery_dir = None

    # ------------------------------------------------------------------
    # Update handling helpers
    def check_for_updates(self) -> None:
        """Poll the configured update endpoint for available releases."""

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
            with urllib.request.urlopen(endpoint, timeout=10.0) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    self.logger.warning(
                        "Update endpoint returned error status",
                        extra={
                            "component": "AppCore",
                            "endpoint": endpoint,
                            "status": status,
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
                extra={
                    "component": "AppCore",
                    "endpoint": endpoint,
                    "payload_type": type(data).__name__,
                },
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

    @staticmethod
    def _current_version() -> str:
        try:  # pragma: no cover - best effort metadata lookup
            from importlib import metadata
        except ImportError:  # pragma: no cover - Python < 3.8
            return "unknown"

        candidates = ["yam-image-processor", "yam_processor"]
        for candidate in candidates:
            try:
                return metadata.version(candidate)
            except metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
        return "unknown"

    def _refresh_allowed_roots(self) -> None:
        candidates = list(self._collect_allowed_root_candidates())
        if not candidates:
            candidates.append(Path.cwd())
        try:
            configure_allowed_roots(candidates)
        except PathValidationError as exc:
            self.logger.warning(
                "Failed to configure allowed roots",
                extra={
                    "component": "AppCore",
                    "error": str(exc),
                    "candidates": [str(path) for path in candidates],
                },
            )
            raise
        self.logger.debug(
            "Allowed roots configured",
            extra={
                "component": "AppCore",
                "roots": [str(path) for path in sanitizer_allowed_roots()],
            },
        )

    def _collect_allowed_root_candidates(self) -> Iterable[Path]:
        candidates: list[Path] = []

        def _append(value: Path | str | None) -> None:
            if value in (None, ""):
                return
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            candidates.append(path)

        for entry in self.config.allowed_roots:
            _append(entry)

        for entry in self._iter_settings_allowed_roots():
            _append(entry)

        _append(self.session_temp_root)
        _append(self.session_pipeline_cache_dir)
        _append(self.session_recovery_dir)
        _append(self.autosave_workspace)

        return candidates

    def _iter_settings_allowed_roots(self) -> Iterable[Path | str]:
        if self.settings_manager is None:
            return ()
        stored = self.settings_manager.get("io/allowed_roots", None)
        return self._coerce_allowed_roots_setting(stored)

    @staticmethod
    def _coerce_allowed_roots_setting(value: Any) -> Iterable[Path | str]:
        if value is None:
            return ()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return (text,)
            if isinstance(parsed, (list, tuple)):
                return tuple(parsed)
            return (text,)
        if isinstance(value, (list, tuple, set)):
            return tuple(value)
        return (value,)

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)


__all__ = ["AppConfiguration", "AppCore", "UpdateDispatcher", "UpdateMetadata"]
