"""Crash recovery coordination for autosave workspaces."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .io_manager import IOManager


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utcnow().strftime("%Y%m%d-%H%M%S")


@dataclass(frozen=True)
class CrashMarker:
    """Description of a crash marker discovered on startup."""

    path: Path
    payload: Dict[str, Any]


@dataclass(frozen=True)
class AutosaveSnapshot:
    """Representation of an autosave payload awaiting user action."""

    workspace: Path
    image_path: Optional[Path]
    metadata_path: Optional[Path]
    metadata: Dict[str, Any]
    backups: Tuple[Path, ...]
    crash_markers: Tuple[CrashMarker, ...]


class RecoveryManager:
    """Inspect autosave workspaces and coordinate crash recovery actions."""

    _SUPPORTED_EXTENSIONS: Tuple[str, ...] = tuple(sorted({
        *(ext.lower() for ext in IOManager.SUPPORTED_EXPORTS.keys()),
        ".npy",
    }))

    def __init__(
        self,
        autosave_directory: Path | str,
        *,
        recovery_root: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._autosave_dir = Path(autosave_directory)
        self._autosave_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger or logging.getLogger(__name__)
        recovery_base = Path(recovery_root) if recovery_root is not None else self._autosave_dir / "recovery"
        self._recovery_base = recovery_base
        self._crash_marker_dir = recovery_base / "crash_markers"
        self._pending_snapshot: Optional[AutosaveSnapshot] = None
        self._crash_markers: Tuple[CrashMarker, ...] = tuple()
        self._crash_detected = False
        self._session_marker_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inspect_startup(self) -> Optional[AutosaveSnapshot]:
        """Inspect the autosave directory for crash artefacts."""

        self._logger.debug(
            "Inspecting autosave workspace for recovery artefacts",
            extra={"workspace": str(self._autosave_dir)},
        )
        self._crash_marker_dir.mkdir(parents=True, exist_ok=True)
        markers = self._load_crash_markers()
        self._crash_markers = tuple(markers)
        self._crash_detected = bool(markers)
        snapshot = self._discover_snapshot()
        self._pending_snapshot = snapshot
        if snapshot is not None:
            self._logger.info(
                "Autosave snapshot discovered",
                extra={
                    "image_path": str(snapshot.image_path) if snapshot.image_path else None,
                    "metadata_path": str(snapshot.metadata_path) if snapshot.metadata_path else None,
                    "backups": len(snapshot.backups),
                },
            )
        self._ensure_session_marker()
        return snapshot

    def has_pending_snapshot(self) -> bool:
        """Return whether an autosave snapshot awaits user input."""

        return self._pending_snapshot is not None

    def pending_snapshot(self) -> Optional[AutosaveSnapshot]:
        """Return the currently pending autosave snapshot, if any."""

        return self._pending_snapshot

    def crash_markers(self) -> Tuple[CrashMarker, ...]:
        """Return crash markers detected during :meth:`inspect_startup`."""

        return self._crash_markers

    def crash_detected(self) -> bool:
        """Return ``True`` when crash markers were discovered on startup."""

        return self._crash_detected

    def confirm_restored(self) -> Optional[AutosaveSnapshot]:
        """Mark the pending autosave snapshot as restored and clean up files."""

        return self._consume_pending(action="restore")

    def discard_pending(self) -> Optional[AutosaveSnapshot]:
        """Discard the pending autosave snapshot and remove artefacts."""

        return self._consume_pending(action="discard")

    def cleanup_crash_markers(self) -> None:
        """Remove any crash markers written for the current session."""

        marker = self._session_marker_path
        if marker is not None and marker.exists():
            try:
                marker.unlink()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self._logger.debug(
                    "Failed to remove crash marker",
                    extra={"path": str(marker), "error": str(exc)},
                )
        self._session_marker_path = None
        self._prune_empty_recovery_dirs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _consume_pending(self, *, action: str) -> Optional[AutosaveSnapshot]:
        snapshot = self._pending_snapshot
        if snapshot is None:
            return None
        self._logger.info(
            "Finalising autosave snapshot",
            extra={"action": action, "workspace": str(snapshot.workspace)},
        )
        self._remove_snapshot_files(snapshot)
        self._pending_snapshot = None
        return snapshot

    def _discover_snapshot(self) -> Optional[AutosaveSnapshot]:
        metadata_path = self._autosave_dir / "autosave.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = self._read_json(metadata_path)
            except Exception as exc:
                self._logger.warning(
                    "Failed to read autosave metadata",
                    extra={"path": str(metadata_path), "error": str(exc)},
                )
                metadata = {}
        image_path = self._resolve_image_path(metadata)
        backups = tuple(self._list_backups())
        has_snapshot = image_path is not None or bool(metadata) or bool(backups)
        if not has_snapshot:
            return None
        metadata_file = metadata_path if metadata_path.exists() else None
        return AutosaveSnapshot(
            workspace=self._autosave_dir,
            image_path=image_path,
            metadata_path=metadata_file,
            metadata=metadata,
            backups=backups,
            crash_markers=self._crash_markers,
        )

    def _resolve_image_path(self, metadata: Mapping[str, Any]) -> Optional[Path]:
        image_info = metadata.get("image") if isinstance(metadata, Mapping) else None
        if isinstance(image_info, Mapping):
            candidate = image_info.get("path")
            if isinstance(candidate, str):
                path = Path(candidate)
                if path.exists():
                    return path
            filename = image_info.get("filename")
            if isinstance(filename, str):
                local_candidate = self._autosave_dir / filename
                if local_candidate.exists():
                    return local_candidate
        for ext in self._SUPPORTED_EXTENSIONS:
            candidate = self._autosave_dir / f"autosave{ext}"
            if candidate.exists():
                return candidate
        return None

    def _list_backups(self) -> Iterable[Path]:
        backups_dir = self._autosave_dir / "backups"
        if not backups_dir.exists():
            return []
        return sorted(path for path in backups_dir.iterdir() if path.is_file())

    def _load_crash_markers(self) -> Iterable[CrashMarker]:
        markers: list[CrashMarker] = []
        directory = self._crash_marker_dir
        if not directory.exists():
            return markers
        for path in sorted(directory.glob("*.json")):
            payload: Dict[str, Any] = {}
            try:
                payload = self._read_json(path)
            except Exception as exc:
                self._logger.warning(
                    "Failed to read crash marker",
                    extra={"path": str(path), "error": str(exc)},
                )
            markers.append(CrashMarker(path, payload))
            try:
                path.unlink()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self._logger.debug(
                    "Failed to remove crash marker",
                    extra={"path": str(path), "error": str(exc)},
                )
        self._prune_empty_recovery_dirs()
        return markers

    def _remove_snapshot_files(self, snapshot: AutosaveSnapshot) -> None:
        for path in (snapshot.image_path, snapshot.metadata_path):
            if path is None:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self._logger.warning(
                    "Failed to remove autosave artefact",
                    extra={"path": str(path), "error": str(exc)},
                )
        backups_dir = self._autosave_dir / "backups"
        for backup in snapshot.backups:
            try:
                backup.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self._logger.warning(
                    "Failed to remove autosave backup",
                    extra={"path": str(backup), "error": str(exc)},
                )
        if backups_dir.exists():
            try:
                if not any(backups_dir.iterdir()):
                    backups_dir.rmdir()
            except OSError:  # pragma: no cover - directory busy
                pass

    def _ensure_session_marker(self) -> None:
        if self._session_marker_path is not None:
            return
        payload = {
            "written_at": _utcnow().isoformat().replace("+00:00", "Z"),
            "workspace": str(self._autosave_dir),
        }
        name = f"session_{_timestamp()}_{uuid.uuid4().hex[:8]}.json"
        path = self._crash_marker_dir / name
        try:
            self._crash_marker_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(path, payload)
        except Exception as exc:  # pragma: no cover - best effort logging
            self._logger.debug(
                "Failed to write crash marker",
                extra={"path": str(path), "error": str(exc)},
            )
            return
        self._session_marker_path = path

    def _prune_empty_recovery_dirs(self) -> None:
        directory = self._crash_marker_dir
        try:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        except OSError:  # pragma: no cover - directory busy
            pass
        base = self._recovery_base
        try:
            if base.exists() and not any(base.iterdir()):
                base.rmdir()
        except OSError:  # pragma: no cover - directory busy
            pass

    def _read_json(self, path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise TypeError("JSON payload must be a mapping")
        return dict(payload)

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        serialised = json.dumps(payload, indent=2, sort_keys=True)
        path.write_text(serialised, encoding="utf-8")


__all__ = [
    "AutosaveSnapshot",
    "CrashMarker",
    "RecoveryManager",
]
