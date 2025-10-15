"""Modal dialog prompting users to review update release information."""

from __future__ import annotations

from enum import Enum
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

if TYPE_CHECKING:
    from yam_processor.core.app_core import UpdateMetadata


class UpdateDialog(QtWidgets.QDialog):
    """Display release notes for an available update."""

    class Action(Enum):
        """Enumeration of user choices within the dialog."""

        ACKNOWLEDGE = "acknowledge"
        INSTALL = "install"

    def __init__(self, metadata: "UpdateMetadata", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._metadata = metadata
        self._selected_action: UpdateDialog.Action = self.Action.ACKNOWLEDGE

        self.setWindowTitle(
            self.tr("Update {version} available").format(version=metadata.version)
        )
        self.setModal(True)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)

        self._build_ui()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> "UpdateMetadata":
        return self._metadata

    @property
    def selected_action(self) -> str:
        return self._selected_action.value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        heading_label = QtWidgets.QLabel(self)
        heading_label.setWordWrap(True)
        heading_label.setText(
            self.tr(
                "<p><strong>Version {version}</strong> is available. Review the "
                "changes below before deciding how to proceed.</p>"
            ).format(version=self._metadata.version)
        )
        layout.addWidget(heading_label)

        self.notes_browser = QtWidgets.QTextBrowser(self)
        self.notes_browser.setObjectName("releaseNotesBrowser")
        self.notes_browser.setOpenExternalLinks(True)
        self.notes_browser.setReadOnly(True)
        notes_text = self._metadata.notes or self.tr("No release notes were provided.")
        # Render plain text to avoid accidental HTML interpretation from metadata.
        self.notes_browser.setPlainText(notes_text)
        layout.addWidget(self.notes_browser)

        if self._metadata.release_notes_url:
            self.release_notes_link = QtWidgets.QLabel(self)
            self.release_notes_link.setObjectName("releaseNotesLink")
            self.release_notes_link.setOpenExternalLinks(True)
            self.release_notes_link.setText(
                self.tr('<a href="{url}">View detailed release notes</a>').format(
                    url=self._metadata.release_notes_url
                )
            )
            layout.addWidget(self.release_notes_link)
        else:
            self.release_notes_link = None

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.acknowledge_button = self.button_box.addButton(
            self.tr("Acknowledge"), QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.install_button = self.button_box.addButton(
            self.tr("Install Update"), QtWidgets.QDialogButtonBox.ActionRole
        )
        self.install_button.setDefault(True)
        self.install_button.setEnabled(self._resolve_install_url() is not None)
        self.button_box.rejected.connect(self.reject)
        self.acknowledge_button.clicked.connect(self._on_acknowledge_clicked)
        self.install_button.clicked.connect(self._on_install_clicked)
        layout.addWidget(self.button_box)

    def _resolve_install_url(self) -> Optional[str]:
        if self._metadata.download_url:
            return self._metadata.download_url
        if self._metadata.release_notes_url:
            return self._metadata.release_notes_url
        return None

    def _on_acknowledge_clicked(self) -> None:
        self._selected_action = self.Action.ACKNOWLEDGE
        self.accept()

    def _on_install_clicked(self) -> None:
        url = self._resolve_install_url()
        if url is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        self._selected_action = self.Action.INSTALL
        self.accept()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def reject(self) -> None:  # pragma: no cover - safety guard
        """Prevent closing the dialog without user acknowledgement."""

        # Intentionally ignore reject requests to keep the dialog modal until one of the
        # explicit actions is chosen.
        return
