"""Persistence helpers providing autosave and backup functionality."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional


_LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utcnow().strftime("%Y%m%d-%H%M%S")


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent))
    try:
        with tmp_file:
            json.dump(payload, tmp_file, indent=2, sort_keys=True)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_file.name, path)
    except Exception:
        try:
            os.unlink(tmp_file.name)
        except FileNotFoundError:
            pass
        raise


def _sidecar_path(destination: Optional[Path]) -> Path:
    if destination is None:
        return Path("autosave.meta.json")
    destination = Path(destination)
    suffix = destination.suffix or ""
    return destination.with_suffix(f"{suffix}.meta.json")


@dataclass
class AutosaveState:
    """Container for the most recent snapshot queued for persistence."""

    pipeline: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutosaveManager:
    """Coordinate periodic autosave snapshots and managed backups."""

    def __init__(
        self,
        autosave_directory: Path,
        *,
        interval_seconds: float = 120.0,
        backup_retention: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._autosave_dir = Path(autosave_directory)
        self._autosave_dir.mkdir(parents=True, exist_ok=True)
        self._interval = max(0.0, float(interval_seconds))
        self._backup_retention = max(0, int(backup_retention))
        self._logger = logger or _LOGGER
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._dirty: bool = False
        self._state: Optional[AutosaveState] = None
        self._project_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Cancel any scheduled autosave operation."""

        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def mark_dirty(self, pipeline: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store ``pipeline`` for persistence and schedule an autosave."""

        snapshot = AutosaveState(pipeline=pipeline, metadata=dict(metadata or {}))
        with self._lock:
            self._state = snapshot
            self._dirty = True
            if self._interval == 0:
                self._logger.debug("Autosave interval is zero; writing immediately")
                self._write_autosave_locked()
            else:
                self._schedule_autosave_locked()

    def save(
        self,
        destination: Optional[Path | str],
        *,
        pipeline: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist the pipeline to ``destination`` creating backups and sidecars."""

        with self._lock:
            if destination is None:
                destination = self._project_path
            if destination is None:
                raise ValueError("A destination path is required for the first save")
            destination = Path(destination)

            state = self._state
            if pipeline is None:
                if state is None:
                    raise ValueError("No pipeline snapshot available for saving")
                pipeline = state.pipeline
                metadata = metadata or state.metadata

        metadata = dict(metadata or {})
        now = _utcnow()
        payload = {
            "pipeline": pipeline,
            "saved_at": _isoformat(now),
        }
        sidecar_payload = {
            "pipeline": pipeline,
            "metadata": metadata,
            "saved_at": _isoformat(now),
            "destination": str(destination),
        }

        try:
            self._create_backup(destination)
            _write_json_atomic(destination, payload)
            _write_json_atomic(_sidecar_path(destination), sidecar_payload)
            self._logger.info("Project saved", extra={"destination": str(destination)})
        except Exception as exc:
            self._logger.error(
                "Failed to save project", extra={"destination": str(destination), "error": str(exc)}
            )
            raise

        with self._lock:
            self._project_path = destination
            self._state = AutosaveState(pipeline=pipeline, metadata=metadata)
            self._dirty = False
        return destination

    def last_known_path(self) -> Optional[Path]:
        with self._lock:
            return self._project_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
            self._write_autosave_locked()

    def _write_autosave_locked(self) -> None:
        if not self._dirty or self._state is None:
            return

        now = _utcnow()
        autosave_path = self._autosave_dir / "autosave.json"
        payload = {
            "pipeline": self._state.pipeline,
            "metadata": self._state.metadata,
            "saved_at": _isoformat(now),
        }
        try:
            _write_json_atomic(autosave_path, payload)
            _write_json_atomic(self._autosave_dir / _sidecar_path(None).name, payload)
            self._logger.info("Autosave snapshot written", extra={"path": str(autosave_path)})
            self._dirty = False
        except Exception as exc:
            self._logger.error(
                "Autosave failed", extra={"path": str(autosave_path), "error": str(exc)}
            )

    def _create_backup(self, destination: Path) -> None:
        destination = Path(destination)
        if not destination.exists():
            return

        backup_dir = destination.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_name = f"{destination.stem}-{_timestamp()}{destination.suffix or ''}"
        backup_path = backup_dir / backup_name
        shutil.copy2(destination, backup_path)
        self._logger.info(
            "Backup created", extra={"source": str(destination), "backup": str(backup_path)}
        )
        if self._backup_retention:
            pattern = f"{destination.stem}-*{destination.suffix or ''}"
            backups = sorted(
                backup_dir.glob(pattern),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for stale in backups[self._backup_retention :]:
                try:
                    stale.unlink()
                except FileNotFoundError:
                    continue
                self._logger.debug(
                    "Backup pruned", extra={"backup": str(stale), "retention": self._backup_retention}
                )


__all__ = ["AutosaveManager"]

