"""Startup dialog presenting available processing environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from plugins.module_base import ModuleStage


@dataclass(frozen=True)
class StartupModuleOption:
    """Describe a selectable processing environment."""

    stage: ModuleStage
    title: str
    description: str = ""


class StartupDialog(QtWidgets.QDialog):
    """Collect startup preferences before launching the main window."""

    def __init__(
        self,
        module_options: Iterable[StartupModuleOption],
        *,
        diagnostics_enabled: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._module_options: Sequence[StartupModuleOption] = tuple(module_options)
        self._selected_stage: ModuleStage | None = None

        self.setWindowTitle(self.tr("Application Startup Configuration"))
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            self.tr(
                "Select the processing environment to launch and choose whether diagnostics"
                " logging should start enabled."
            )
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._button_group = QtWidgets.QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._button_group.buttonClicked[int].connect(self._update_selected_stage)

        if self._module_options:
            module_group = QtWidgets.QGroupBox(self.tr("Available environments"))
            module_layout = QtWidgets.QVBoxLayout(module_group)
            for index, option in enumerate(self._module_options):
                button = QtWidgets.QRadioButton(option.title, module_group)
                if option.description:
                    button.setToolTip(option.description)
                    button.setStatusTip(option.description)
                self._button_group.addButton(button, index)
                module_layout.addWidget(button)
                if index == 0:
                    button.setChecked(True)
            module_layout.addStretch(1)
            layout.addWidget(module_group)
        else:
            empty_label = QtWidgets.QLabel(
                self.tr("No processing environments are currently available.")
            )
            empty_label.setAlignment(QtCore.Qt.AlignHCenter)
            empty_label.setObjectName("startupDialogEmptyLabel")
            layout.addWidget(empty_label)

        self._diagnostics_checkbox = QtWidgets.QCheckBox(
            self.tr("Enable diagnostics logging"), self
        )
        self._diagnostics_checkbox.setChecked(bool(diagnostics_enabled))
        self._diagnostics_checkbox.setToolTip(
            self.tr(
                "Diagnostics logging increases verbosity and unlocks additional telemetry "
                "controls."
            )
        )
        layout.addWidget(self._diagnostics_checkbox)

        layout.addStretch(1)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(self._module_options))
        layout.addWidget(button_box)

        self._selected_stage = self._resolve_selected_stage()

    def _update_selected_stage(self, _: int) -> None:
        self._selected_stage = self._resolve_selected_stage()

    def _resolve_selected_stage(self) -> ModuleStage | None:
        checked_id = self._button_group.checkedId()
        if checked_id == -1:
            return None
        if 0 <= checked_id < len(self._module_options):
            return self._module_options[checked_id].stage
        return None

    def accept(self) -> None:  # type: ignore[override]
        self._selected_stage = self._resolve_selected_stage()
        super().accept()

    @property
    def selected_stage(self) -> ModuleStage | None:
        """Return the processing stage selected by the user, if any."""

        return self._selected_stage

    @property
    def diagnostics_enabled(self) -> bool:
        """Return whether diagnostics logging should be enabled."""

        return self._diagnostics_checkbox.isChecked()


__all__ = ["StartupDialog", "StartupModuleOption"]
