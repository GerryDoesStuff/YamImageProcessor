"""Startup dialog presenting available processing environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from core.settings import SettingsManager
from plugins.module_base import ModuleStage


_DEFAULT_SELECTION_KEY = "ui/startup/selected_stages"


@dataclass(frozen=True)
class StartupModuleOption:
    """Describe a selectable processing environment."""

    stage: ModuleStage
    title: str
    description: str = ""
    enabled_by_default: bool = True


class StartupDialog(QtWidgets.QDialog):
    """Collect startup preferences before launching the main window."""

    def __init__(
        self,
        module_options: Iterable[StartupModuleOption],
        *,
        diagnostics_enabled: bool = False,
        settings: SettingsManager | None = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._module_options: Sequence[StartupModuleOption] = tuple(module_options)
        self._settings = settings
        self._selected_stages: list[ModuleStage] = []

        self.setWindowTitle(self.tr("Application Startup Configuration"))
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            self.tr(
                "Select the processing environments to launch and choose whether diagnostics"
                " logging should start enabled."
            )
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        if self._module_options:
            module_group = QtWidgets.QGroupBox(self.tr("Available environments"))
            module_layout = QtWidgets.QVBoxLayout(module_group)

            self._stage_list = QtWidgets.QListWidget(module_group)
            self._stage_list.setObjectName("startupStageList")
            self._stage_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            self._stage_list.itemChanged.connect(self._sync_selected_stages)

            default_selection = self._load_default_selection()

            for option in self._module_options:
                item = QtWidgets.QListWidgetItem(option.title, self._stage_list)
                item.setFlags(
                    item.flags()
                    | QtCore.Qt.ItemIsUserCheckable
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsEnabled
                )
                item.setData(QtCore.Qt.UserRole, option.stage)
                if option.description:
                    item.setToolTip(option.description)
                    item.setStatusTip(option.description)
                is_checked = option.stage.name in default_selection
                if not default_selection:
                    is_checked = option.enabled_by_default
                item.setCheckState(
                    QtCore.Qt.Checked if is_checked else QtCore.Qt.Unchecked
                )
                self._stage_list.addItem(item)

            module_layout.addWidget(self._stage_list)
            layout.addWidget(module_group)
        else:
            self._stage_list = None
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
        self._ok_button = button_box.button(QtWidgets.QDialogButtonBox.Ok)
        self._ok_button.setEnabled(bool(self._module_options))
        layout.addWidget(button_box)

        self._sync_selected_stages()

    def _load_default_selection(self) -> set[str]:
        if self._settings is None:
            return set()
        payload = self._settings.get(_DEFAULT_SELECTION_KEY, "")
        if not payload:
            return set()
        try:
            data = json.loads(str(payload))
        except (TypeError, ValueError):
            return set()
        if isinstance(data, list):
            return {str(item) for item in data}
        return set()

    def _sync_selected_stages(self) -> None:
        self._selected_stages = self._collect_selected_stages()
        if hasattr(self, "_ok_button") and self._ok_button is not None:
            self._ok_button.setEnabled(bool(self._selected_stages))

    def _collect_selected_stages(self) -> list[ModuleStage]:
        selections: list[ModuleStage] = []
        if self._stage_list is None:
            return selections
        for index in range(self._stage_list.count()):
            item = self._stage_list.item(index)
            if item is None:
                continue
            if item.checkState() != QtCore.Qt.Checked:
                continue
            stage = item.data(QtCore.Qt.UserRole)
            if isinstance(stage, ModuleStage):
                selections.append(stage)
        return selections

    def accept(self) -> None:  # type: ignore[override]
        self._sync_selected_stages()
        if self._settings is not None:
            payload = json.dumps([stage.name for stage in self._selected_stages])
            self._settings.set(_DEFAULT_SELECTION_KEY, payload)
        super().accept()

    @property
    def selected_stages(self) -> Sequence[ModuleStage]:
        """Return the ordered list of selected processing stages."""

        return tuple(self._selected_stages)

    @property
    def selected_stage(self) -> ModuleStage | None:
        """Return the first selected processing stage, if any."""

        return self._selected_stages[0] if self._selected_stages else None

    @property
    def diagnostics_enabled(self) -> bool:
        """Return whether diagnostics logging should be enabled."""

        return self._diagnostics_checkbox.isChecked()


__all__ = ["StartupDialog", "StartupModuleOption"]
