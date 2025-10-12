"""User interface layer utilities.

Applications should create a :class:`~yam_processor.ui.main_window.MainWindow`
instance during startup, optionally providing a
:class:`~yam_processor.ui.pipeline_controller.PipelineController` to wire in
pipeline behaviours before showing the window.
"""

from .diagnostics_panel import DiagnosticsPanel
from .dialogs import ParameterDialog, ParameterSpec, ParameterType, PreviewWidget
from .error_dialog import ErrorDialog
from .main_window import MainWindow
from .pipeline_controller import PipelineController

__all__ = [
    "MainWindow",
    "DiagnosticsPanel",
    "ParameterDialog",
    "ParameterSpec",
    "ParameterType",
    "PipelineController",
    "PreviewWidget",
    "ErrorDialog",
]
