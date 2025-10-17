"""Reusable startup dialog that allows toggling diagnostics and modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from PyQt5 import QtCore, QtWidgets

from core.app_core import AppCore
from plugins.module_base import ModuleStage


@dataclass
class _ModuleEntry:
    identifier: str
    title: str
    description: str
    default_checked: bool


class StartupDialog(QtWidgets.QDialog):
    """Collect startup preferences before launching the main window."""

    def __init__(
        self,
        app_core: AppCore,
        stage: ModuleStage,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_core = app_core
        self._stage = stage
        self._module_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}

        self.setWindowTitle(self.tr("Application Startup Configuration"))
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            self.tr(
                "Select the {stage} modules to enable and choose whether diagnostics logging"
                " should start enabled."
            ).format(stage=self._format_stage_name(stage))
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        module_entries = list(self._build_module_entries())
        if module_entries:
            module_group = QtWidgets.QGroupBox(
                self.tr("{stage} modules").format(
                    stage=self._format_stage_name(stage)
                )
            )
            module_layout = QtWidgets.QVBoxLayout(module_group)
            for entry in module_entries:
                checkbox = QtWidgets.QCheckBox(entry.title, module_group)
                checkbox.setChecked(entry.default_checked)
                if entry.description:
                    checkbox.setToolTip(entry.description)
                    checkbox.setStatusTip(entry.description)
                module_layout.addWidget(checkbox)
                self._module_checkboxes[entry.identifier] = checkbox
            module_layout.addStretch(1)
            layout.addWidget(module_group)
        else:
            empty_label = QtWidgets.QLabel(
                self.tr("No modules are currently registered for this stage.")
            )
            empty_label.setAlignment(QtCore.Qt.AlignHCenter)
            empty_label.setObjectName("startupDialogEmptyLabel")
            layout.addWidget(empty_label)

        diagnostics_enabled = self._initial_diagnostics_state()
        self._diagnostics_checkbox = QtWidgets.QCheckBox(
            self.tr("Enable diagnostics logging"), self
        )
        self._diagnostics_checkbox.setChecked(diagnostics_enabled)
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
        layout.addWidget(button_box)

    @staticmethod
    def _format_stage_name(stage: ModuleStage) -> str:
        return stage.value.replace("_", " ").title()

    def _build_module_entries(self) -> Iterable[_ModuleEntry]:
        modules = sorted(
            self._app_core.iter_modules(self._stage),
            key=lambda module: module.metadata.title.lower(),
        )
        for module in modules:
            metadata = module.metadata
            yield _ModuleEntry(
                identifier=metadata.identifier,
                title=metadata.title,
                description=metadata.description,
                default_checked=self._app_core.module_enabled(
                    self._stage, metadata.identifier
                ),
            )

    def _initial_diagnostics_state(self) -> bool:
        settings = getattr(self._app_core, "settings_manager", None)
        if settings is not None:
            if settings.contains("diagnostics/enabled"):
                return settings.get_bool("diagnostics/enabled", False)
        return self._app_core.diagnostics_enabled

    def accept(self) -> None:  # type: ignore[override]
        for identifier, checkbox in self._module_checkboxes.items():
            self._app_core.set_module_enabled(
                self._stage, identifier, checkbox.isChecked()
            )
        self._app_core.set_diagnostics_enabled(
            self._diagnostics_checkbox.isChecked(), persist=True
        )
        super().accept()


__all__ = ["StartupDialog"]
