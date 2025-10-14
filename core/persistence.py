"""Autosave and persistence helpers built atop :class:`IOManager`."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .io_manager import IOManager, SaveResult
from .recovery import RecoveryManager
from .settings import SettingsManager


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utcnow().strftime("%Y%m%d-%H%M%S")


@dataclass
class AutosavePayload:
    """Container describing the most recent state queued for persistence."""

    image: np.ndarray
    pipeline: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_inputs(
        cls,
        image: np.ndarray,
        pipeline: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AutosavePayload":
        payload = cls(
            image=np.array(image, copy=True),
            pipeline=dict(pipeline),
            metadata=dict(metadata or {}),
        )
        return payload


class AutosaveManager:
    """Coordinate scheduled autosaves and managed project persistence."""

    def __init__(
        self,
        settings: SettingsManager,
        io_manager: IOManager,
        autosave_directory: Path,
        *,
        interval_seconds: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        recovery_manager: Optional[RecoveryManager] = None,
    ) -> None:
        self._settings = settings
        self._io_manager = io_manager
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._payload: Optional[AutosavePayload] = None
        self._dirty = False
        self._last_result: Optional[SaveResult] = None
        self._project_path: Optional[Path] = None
        self._autosave_dir = Path(autosave_directory)
        self._autosave_dir.mkdir(parents=True, exist_ok=True)
        self._recovery = recovery_manager

        self._enabled = self._settings.autosave_enabled()
        self._interval = self._resolve_interval(interval_seconds)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def shutdown(self) -> None:
        """Cancel any scheduled autosave timers."""

        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    # ------------------------------------------------------------------
    # Public API
    def refresh_preferences(self) -> None:
        """Re-read autosave preferences from the settings backend."""

        with self._lock:
            self._enabled = self._settings.autosave_enabled()
            self._interval = self._resolve_interval(None)

    def mark_dirty(
        self,
        image: np.ndarray,
        pipeline: Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Store ``image`` and ``pipeline`` and schedule an autosave."""

        snapshot = AutosavePayload.from_inputs(image, pipeline, metadata)
        with self._lock:
            self._payload = snapshot
            self._dirty = True
            self._enabled = self._settings.autosave_enabled()
            self._interval = self._resolve_interval(None)
            if not self._enabled:
                self._logger.debug("Autosave disabled; snapshot retained without scheduling")
                return
            if self._interval == 0:
                self._logger.debug("Autosave interval is zero; writing immediately")
                self._write_autosave_locked(create_backup=True)
            else:
                self._schedule_autosave_locked()

    def save(
        self,
        destination: Path | str | None,
        *,
        image: Optional[np.ndarray] = None,
        pipeline: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        create_backup: bool = True,
    ) -> SaveResult:
        """Persist the provided payload to ``destination`` using the IO manager."""

        with self._lock:
            if destination is None:
                destination = self._project_path
            if destination is None:
                raise ValueError("A destination path must be provided for the initial save")
            destination_path = Path(destination)

            payload = self._payload
            if image is None or pipeline is None:
                if payload is None:
                    raise ValueError("No autosave payload available for persistence")
                if image is None:
                    image = payload.image
                if pipeline is None:
                    pipeline = payload.pipeline
                if metadata is None:
                    metadata = payload.metadata

            metadata_dict = dict(metadata or {})
            marker = None
            if self._recovery is not None:
                marker = self._recovery.begin_guarded_write(
                    reason="project_save",
                    destination=destination_path,
                    metadata={"create_backup": bool(create_backup)},
                )
            try:
                result = self._io_manager.save_image(
                    destination_path,
                    image,
                    metadata=metadata_dict,
                    pipeline=dict(pipeline),
                    settings_snapshot=self._settings_snapshot(),
                    create_backup=create_backup,
                    backup_retention=self._settings.autosave_backup_retention(),
                )
            except Exception:
                if self._recovery is not None:
                    self._recovery.complete_guarded_write(marker, success=False)
                raise
            if self._recovery is not None:
                self._recovery.complete_guarded_write(marker, success=True)
                self._recovery.notify_autosave_cleared()
            self._project_path = result.image_path
            self._payload = AutosavePayload.from_inputs(image, pipeline, metadata_dict)
            self._dirty = False
            self._last_result = result
            return result

    def last_result(self) -> Optional[SaveResult]:
        with self._lock:
            return self._last_result

    def last_project_path(self) -> Optional[Path]:
        with self._lock:
            return self._project_path

    # ------------------------------------------------------------------
    # Internal helpers
    def _resolve_interval(self, override: Optional[float]) -> float:
        if override is not None:
            return max(0.0, float(override))
        return max(0.0, float(self._settings.autosave_interval()))

    def _schedule_autosave_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        timer = threading.Timer(self._interval, self._autosave_callback)
        timer.daemon = True
        timer.start()
        self._timer = timer
        self._logger.debug("Autosave scheduled", extra={"interval": self._interval})

    def _autosave_callback(self) -> None:
        with self._lock:
            self._timer = None
            self._write_autosave_locked(create_backup=True)

    def _write_autosave_locked(self, *, create_backup: bool) -> None:
        if not self._dirty or self._payload is None:
            return

        destination = self._autosave_dir / f"autosave{self._io_manager.default_format}"
        timestamp = _timestamp()
        metadata = dict(self._payload.metadata)
        metadata.setdefault("autosave", {})
        metadata["autosave"].update({
            "written_at": _utcnow().isoformat().replace("+00:00", "Z"),
            "source": "autosave",
            "timestamp": timestamp,
        })
        marker = None
        if self._recovery is not None:
            marker = self._recovery.begin_guarded_write(
                reason="autosave",
                destination=destination,
                metadata={"timestamp": timestamp},
            )
        try:
            result = self._io_manager.save_image(
                destination,
                self._payload.image,
                metadata=metadata,
                pipeline=dict(self._payload.pipeline),
                settings_snapshot=self._settings_snapshot(),
                create_backup=create_backup,
                backup_retention=self._settings.autosave_backup_retention(),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error(
                "Failed to write autosave snapshot",
                extra={"destination": str(destination), "error": str(exc)},
            )
            if self._recovery is not None:
                self._recovery.complete_guarded_write(marker, success=False)
            return

        self._dirty = False
        self._last_result = result
        if self._recovery is not None:
            self._recovery.complete_guarded_write(marker, success=True)
            self._recovery.notify_autosave_written()
        self._logger.info(
            "Autosave snapshot written",
            extra={"destination": str(result.image_path)},
        )

    def _settings_snapshot(self) -> Dict[str, object]:
        return self._settings.snapshot()


__all__ = ["AutosaveManager", "AutosavePayload"]
