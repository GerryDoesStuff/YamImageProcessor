"""Crash recovery coordination for autosave and pipeline artefacts."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PyQt5 import QtCore, QtWidgets  # type: ignore


@dataclass(slots=True)
class AutosaveArtefact:
    """Container describing an autosave payload discovered on startup."""

    payload_path: Path
    metadata_path: Optional[Path]
    payload: dict[str, Any]
    metadata: dict[str, Any]


class RecoveryManager:
    """Inspect autosave/backup artefacts and coordinate crash recovery flows."""

    def __init__(
        self,
        autosave_directory: Path,
        *,
        crash_marker_root: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._autosave_dir = Path(autosave_directory)
        self._autosave_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger or logging.getLogger(__name__)
        self._crash_marker_root = Path(crash_marker_root) if crash_marker_root else None
        self._pending_autosave: Optional[AutosaveArtefact] = None
        self._restored_payload: Optional[dict[str, Any]] = None
        self._restored_metadata: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inspect_startup(self, *, parent: Optional[QtWidgets.QWidget] = None) -> Optional[dict[str, Any]]:
        """Inspect the autosave directory and prompt the user if possible."""

        self._pending_autosave = self._load_autosave()
        if self._pending_autosave is None:
            self._logger.debug("No autosave artefacts detected during startup")
            return None
        return self._prompt_pending(parent=parent)

    def prompt_pending(self, *, parent: Optional[QtWidgets.QWidget] = None) -> Optional[dict[str, Any]]:
        """Prompt for any autosave previously discovered via :meth:`inspect_startup`."""

        return self._prompt_pending(parent=parent)

    def restored_payload(self) -> Optional[dict[str, Any]]:
        """Return the payload restored from the most recent autosave."""

        return self._restored_payload

    def restored_metadata(self) -> Optional[dict[str, Any]]:
        """Return metadata associated with the restored autosave payload."""

        return self._restored_metadata

    def has_pending_autosave(self) -> bool:
        """Return whether an autosave snapshot is awaiting user action."""

        return self._pending_autosave is not None

    def discard_pending(self) -> None:
        """Discard any pending autosave artefact without prompting."""

        artefact = self._pending_autosave
        if artefact is None:
            return
        self._logger.info("Discarding pending autosave artefact without user prompt")
        self._remove_autosave(artefact)
        self._pending_autosave = None

    def cleanup_crash_markers(self) -> None:
        """Remove stale crash recovery markers written by pipeline failures."""

        root = self._crash_marker_root
        if root is None:
            return
        if not root.exists():
            return
        for child in root.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self._logger.debug("Failed to remove crash artefact", extra={"path": str(child), "error": str(exc)})
        try:
            shutil.rmtree(root, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            self._logger.debug("Failed to remove crash marker root", extra={"path": str(root), "error": str(exc)})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prompt_pending(self, *, parent: Optional[QtWidgets.QWidget] = None) -> Optional[dict[str, Any]]:
        artefact = self._pending_autosave
        if artefact is None:
            return None
        if QtWidgets.QApplication.instance() is None:
            self._logger.warning(
                "Autosave artefact pending but no QApplication is available for prompting"
            )
            return None
        decision = self._prompt_user(artefact, parent=parent)
        if decision == "restore":
            self._logger.info(
                "Restoring autosave snapshot", extra={"path": str(artefact.payload_path)}
            )
            self._restored_payload = artefact.payload
            self._restored_metadata = artefact.metadata
            self._remove_autosave(artefact)
            self._pending_autosave = None
            return artefact.payload
        self._logger.info(
            "Discarding autosave snapshot", extra={"path": str(artefact.payload_path)}
        )
        self._remove_autosave(artefact)
        self._pending_autosave = None
        return None

    def _load_autosave(self) -> Optional[AutosaveArtefact]:
        payload_path = self._autosave_dir / "autosave.json"
        if not payload_path.exists():
            return None
        metadata_path = self._autosave_dir / "autosave.meta.json"
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._logger.error(
                "Failed to read autosave payload", extra={"path": str(payload_path), "error": str(exc)}
            )
            return None
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._logger.warning(
                    "Failed to read autosave metadata", extra={"path": str(metadata_path), "error": str(exc)}
                )
        return AutosaveArtefact(payload_path, metadata_path if metadata_path.exists() else None, payload, metadata)

    def _prompt_user(
        self,
        artefact: AutosaveArtefact,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> str:
        window_title = self._tr("Autosave Available")
        saved_at = artefact.metadata.get("saved_at") or artefact.payload.get("saved_at")
        if saved_at:
            message = self._tr("An autosave from {timestamp} was found. Restore it?").format(
                timestamp=saved_at
            )
        else:
            message = self._tr("An autosave snapshot was found. Restore it?")
        details: list[str] = []
        destination = artefact.metadata.get("destination")
        if destination:
            details.append(self._tr("Original location: {path}").format(path=destination))
        count_backups = len(list((self._autosave_dir / "backups").glob("*")))
        if count_backups:
            details.append(
                self._tr("{count} backup file(s) are also available").format(count=count_backups)
            )

        box = QtWidgets.QMessageBox(parent)
        box.setWindowTitle(window_title)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(message)
        if details:
            box.setInformativeText("\n".join(details))
        restore_button = box.addButton(self._tr("&Restore"), QtWidgets.QMessageBox.AcceptRole)
        box.addButton(self._tr("&Discard"), QtWidgets.QMessageBox.DestructiveRole)
        box.setDefaultButton(restore_button)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == restore_button:
            return "restore"
        return "discard"

    def _remove_autosave(self, artefact: AutosaveArtefact) -> None:
        for path in (artefact.payload_path, artefact.metadata_path):
            if not path:
                continue
            try:
                path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self._logger.debug(
                    "Failed to remove autosave artefact", extra={"path": str(path), "error": str(exc)}
                )

    @staticmethod
    def _tr(text: str) -> str:
        return QtCore.QCoreApplication.translate("RecoveryManager", text)


__all__ = ["RecoveryManager"]
