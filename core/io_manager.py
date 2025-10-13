"""Persistence helpers for image import/export with metadata sidecars."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from .settings import SettingsManager


class PersistenceError(RuntimeError):
    """Raised when an image or metadata payload cannot be persisted."""


@dataclass(frozen=True)
class SaveResult:
    """Details about a completed save operation."""

    image_path: Path
    metadata_path: Path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utcnow().strftime("%Y%m%d-%H%M%S")


class IOManager:
    """Coordinate loading and saving image arrays alongside metadata."""

    DEFAULT_FORMAT = ".png"
    DEFAULT_METADATA_SCHEMA = "yam.image-metadata.v1"
    DEFAULT_BACKUP_RETENTION = 5
    SUPPORTED_EXPORTS: Dict[str, str] = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tiff": "TIFF",
        ".bmp": "BMP",
        ".npy": "NPY",
    }

    def __init__(self, settings: SettingsManager | Mapping[str, Any] | None = None) -> None:
        self._settings_source = settings
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Preference helpers
    @property
    def default_format(self) -> str:
        value = self._get_setting("io/default_format", self.DEFAULT_FORMAT)
        return self._normalise_extension(str(value))

    @property
    def metadata_schema(self) -> str:
        value = self._get_setting("io/metadata_schema", self.DEFAULT_METADATA_SCHEMA)
        text = str(value).strip()
        return text or self.DEFAULT_METADATA_SCHEMA

    @property
    def backup_retention(self) -> int:
        value = self._get_setting(
            "autosave/backup_retention", self.DEFAULT_BACKUP_RETENTION
        )
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return self.DEFAULT_BACKUP_RETENTION

    def export_preferences(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of persistence preferences."""

        return {
            "io/default_format": self.default_format,
            "io/metadata_schema": self.metadata_schema,
        }

    # ------------------------------------------------------------------
    # Public API
    def save_image(
        self,
        destination: str | os.PathLike[str],
        image: np.ndarray,
        *,
        metadata: Mapping[str, Any] | None = None,
        pipeline: Mapping[str, Any] | None = None,
        settings_snapshot: Mapping[str, Any] | None = None,
        format_hint: str | None = None,
        create_backup: bool = True,
        backup_retention: Optional[int] = None,
    ) -> SaveResult:
        """Persist ``image`` and emit a JSON metadata sidecar."""

        path = Path(destination)
        path, ext = self._resolve_destination(path, format_hint)
        path.parent.mkdir(parents=True, exist_ok=True)

        if create_backup:
            if backup_retention is None:
                retention = self.backup_retention
            else:
                try:
                    retention = max(0, int(backup_retention))
                except (TypeError, ValueError):
                    retention = self.backup_retention
            self._create_backup(path, retention)

        if ext == ".npy":
            np.save(path, image, allow_pickle=False)
        else:
            success = cv2.imwrite(str(path), image)
            if not success:
                raise PersistenceError(f"Failed to save image to '{path}'.")

        metadata_path = self._write_metadata_sidecar(
            path,
            metadata=self._ensure_mapping(metadata),
            pipeline=self._ensure_mapping(pipeline),
            settings=self._ensure_mapping(settings_snapshot),
            fmt=self.SUPPORTED_EXPORTS[ext],
        )
        return SaveResult(image_path=path, metadata_path=metadata_path)

    def load_image(
        self,
        source: str | os.PathLike[str],
        *,
        read_metadata: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Load an image array and optional metadata sidecar."""

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        ext = self._normalise_extension(path.suffix)
        if ext not in self.SUPPORTED_EXPORTS:
            raise PersistenceError(f"Unsupported image format '{path.suffix}'.")

        if ext == ".npy":
            image = np.load(path, allow_pickle=False)
        else:
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if array is None:
                raise PersistenceError(f"Failed to load image from '{path}'.")
            image = array

        metadata_payload: Optional[Dict[str, Any]] = None
        if read_metadata:
            metadata_path = self._sidecar_path(path)
            if metadata_path.exists():
                try:
                    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._logger.warning(
                        "Failed to read metadata sidecar",
                        extra={"path": str(metadata_path), "error": str(exc)},
                    )
        return image, metadata_payload

    # ------------------------------------------------------------------
    # Internal helpers
    def _get_setting(self, key: str, default: Any) -> Any:
        source = self._settings_source
        if isinstance(source, SettingsManager):
            return source.get(key, default)
        if isinstance(source, Mapping):
            return source.get(key, default)
        return default

    def _normalise_extension(self, extension: str) -> str:
        ext = extension.lower()
        if not ext.startswith(".") and ext:
            ext = f".{ext}"
        if ext == ".jpeg":
            ext = ".jpg"
        if ext and ext not in self.SUPPORTED_EXPORTS:
            raise PersistenceError(f"Unsupported image format '{extension}'.")
        return ext or self.DEFAULT_FORMAT

    def _resolve_destination(
        self, path: Path, format_hint: str | None
    ) -> Tuple[Path, str]:
        ext = path.suffix
        if ext:
            ext = self._normalise_extension(ext)
            path = path.with_suffix(ext)
        else:
            resolved = self.default_format
            if format_hint:
                resolved = self._normalise_extension(format_hint)
            path = path.with_suffix(resolved)
            ext = resolved
        return path, ext

    def _sidecar_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def _ensure_mapping(self, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
        raise TypeError("Metadata payloads must be mappings.")

    def _write_metadata_sidecar(
        self,
        image_path: Path,
        *,
        metadata: Dict[str, Any],
        pipeline: Dict[str, Any],
        settings: Dict[str, Any],
        fmt: str,
    ) -> Path:
        payload: Dict[str, Any] = {
            "schema": self.metadata_schema,
            "image": {
                "filename": image_path.name,
                "path": str(image_path),
                "format": fmt,
            },
            "metadata": metadata,
            "pipeline": pipeline,
            "settings": settings,
        }
        try:
            serialised = json.dumps(payload, indent=2, sort_keys=True)
        except TypeError as exc:
            raise PersistenceError(f"Metadata payload is not JSON serialisable: {exc}") from exc

        metadata_path = self._sidecar_path(image_path)
        fd, temp_path = tempfile.mkstemp(dir=str(metadata_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialised)
            Path(temp_path).replace(metadata_path)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:  # pragma: no cover - already moved
                pass
        return metadata_path

    def _create_backup(self, path: Path, retention: int) -> None:
        if retention <= 0 or not path.exists():
            return

        backup_dir = path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _timestamp()
        suffix = path.suffix or ""
        backup_name = f"{path.stem}-{timestamp}{suffix}"
        backup_path = backup_dir / backup_name
        try:
            shutil.copy2(path, backup_path)
        except FileNotFoundError:
            return

        metadata_path = self._sidecar_path(path)
        if metadata_path.exists():
            metadata_backup = backup_dir / f"{metadata_path.stem}-{timestamp}{metadata_path.suffix}"
            try:
                shutil.copy2(metadata_path, metadata_backup)
            except FileNotFoundError:
                pass

        self._logger.debug(
            "Backup created",
            extra={"source": str(path), "backup": str(backup_path)},
        )

        if retention:
            pattern = f"{path.stem}-*{suffix}"
            backups = sorted(
                backup_dir.glob(pattern),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            sidecar_suffix = self._sidecar_path(path).suffix
            for stale in backups[retention:]:
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
                metadata_candidate = backup_dir / f"{stale.stem}{sidecar_suffix}"
                try:
                    metadata_candidate.unlink()
                except FileNotFoundError:
                    pass


__all__ = ["IOManager", "SaveResult", "PersistenceError"]
