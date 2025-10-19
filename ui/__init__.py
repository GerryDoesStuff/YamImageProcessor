"""Qt user interface components for the standalone utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PyQt5 import QtWidgets


class ModulePane(QtWidgets.QWidget, ABC):
    """Interface describing the widgets embedded within the unified shell."""

    @abstractmethod
    def on_activated(self) -> None:
        """Invoked when the pane becomes the active tab."""

    @abstractmethod
    def on_deactivated(self) -> None:
        """Invoked when the pane is no longer the active tab."""

    @abstractmethod
    def load_image(self) -> None:
        """Request that the pane prompts the user to load an image."""

    @abstractmethod
    def save_outputs(self) -> None:
        """Request that the pane exports any generated outputs."""

    @abstractmethod
    def update_pipeline_summary(self) -> None:
        """Request that the pane refreshes any pipeline summary views."""

    @abstractmethod
    def set_diagnostics_visible(self, visible: bool) -> None:
        """Notify the pane that diagnostics visibility has changed."""

    @abstractmethod
    def refresh_menus(self) -> None:
        """Request that the pane rebuilds the host window menu bar."""

    @abstractmethod
    def teardown(self) -> None:
        """Allow the pane to perform cleanup prior to application shutdown."""


__all__ = ["ModulePane"]
