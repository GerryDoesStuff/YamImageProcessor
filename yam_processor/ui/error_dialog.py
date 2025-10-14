"""Shared Qt dialog for presenting recoverable errors to the user."""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from PyQt5 import QtCore, QtWidgets  # type: ignore


class ErrorDialog(QtWidgets.QDialog):
    """Modal dialog that displays an error message and traceback."""

    def __init__(
        self,
        message: str,
        traceback_text: str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        metadata: Optional[Mapping[str, object]] = None,
        window_title: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self._message = message
        self._traceback_text = traceback_text
        self._metadata = dict(metadata or {})

        if window_title is None:
            window_title = self.tr("An unexpected error occurred")
        self.setWindowTitle(window_title)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        icon_label = QtWidgets.QLabel(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical)
        icon_label.setPixmap(icon.pixmap(32, 32))

        text_layout = QtWidgets.QHBoxLayout()
        text_layout.setSpacing(10)
        text_layout.addWidget(icon_label, 0, QtCore.Qt.AlignTop)

        message_label = QtWidgets.QLabel(self._message, self)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        text_layout.addWidget(message_label, 1)

        layout.addLayout(text_layout)

        if self._metadata:
            metadata_group = QtWidgets.QGroupBox(self.tr("Details"), self)
            metadata_layout = QtWidgets.QFormLayout(metadata_group)
            metadata_layout.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            metadata_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            metadata_layout.setSpacing(6)
            for key, value in sorted(self._metadata.items(), key=lambda item: str(item[0])):
                label = QtWidgets.QLabel(str(key), metadata_group)
                label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                value_label = QtWidgets.QLabel(str(value), metadata_group)
                value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                value_label.setWordWrap(True)
                metadata_layout.addRow(label, value_label)
            layout.addWidget(metadata_group)

        traceback_group = QtWidgets.QGroupBox(self.tr("Technical information"), self)
        traceback_layout = QtWidgets.QVBoxLayout(traceback_group)
        traceback_layout.setContentsMargins(6, 6, 6, 6)
        traceback_layout.setSpacing(4)
        self._trace_edit = QtWidgets.QPlainTextEdit(traceback_group)
        self._trace_edit.setReadOnly(True)
        self._trace_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._trace_edit.setPlainText(self._traceback_text)
        self._trace_edit.setMinimumHeight(180)
        traceback_layout.addWidget(self._trace_edit)
        layout.addWidget(traceback_group, 1)

        self._button_box = QtWidgets.QDialogButtonBox(self)
        self._copy_button = self._button_box.addButton(
            self.tr("Copy details"), QtWidgets.QDialogButtonBox.ActionRole
        )
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        close_button = self._button_box.addButton(
            QtWidgets.QDialogButtonBox.Close
        )
        close_button.clicked.connect(self.accept)
        self._status_label = QtWidgets.QLabel(self)
        self._status_label.setWordWrap(True)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)
        layout.addWidget(self._button_box)

    @QtCore.pyqtSlot()
    def copy_to_clipboard(self) -> None:
        """Copy the dialog contents to the system clipboard."""

        clipboard = QtWidgets.QApplication.clipboard()
        sections = [self._message]
        if self._metadata:
            sections.append("\n".join(f"{key}: {value}" for key, value in self._metadata.items()))
        sections.append(self._traceback_text)
        clipboard.setText("\n\n".join(section for section in sections if section))

    def add_action_button(
        self,
        label: str,
        *,
        role: QtWidgets.QDialogButtonBox.ButtonRole = QtWidgets.QDialogButtonBox.ActionRole,
        callback: Optional[Callable[[], None]] = None,
    ) -> QtWidgets.QAbstractButton:
        """Add a custom action button to the dialog."""

        button = self._button_box.addButton(label, role)
        if callback is not None:
            button.clicked.connect(callback)
        return button

    def set_status_message(self, text: str, *, error: bool = False) -> None:
        """Display a transient status message beneath the actions."""

        if not text:
            self._status_label.clear()
            self._status_label.setVisible(False)
            return
        self._status_label.setText(text)
        if error:
            self._status_label.setStyleSheet("color: #b00020;")
        else:
            self._status_label.setStyleSheet("")
        self._status_label.setVisible(True)

    @classmethod
    def present(
        cls,
        message: str,
        traceback_text: str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        metadata: Optional[Mapping[str, object]] = None,
        window_title: Optional[str] = None,
    ) -> "ErrorDialog":
        dialog = cls(
            message,
            traceback_text,
            parent=parent,
            metadata=metadata,
            window_title=window_title,
        )
        dialog.exec_()
        return dialog


__all__ = ["ErrorDialog"]
