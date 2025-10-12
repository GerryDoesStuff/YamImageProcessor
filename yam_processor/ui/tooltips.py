"""Tooltip helpers shared across Yam Image Processor widgets."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, TYPE_CHECKING

from PyQt5 import QtCore  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .pipeline_controller import PipelineController


def _join_sections(sections: Iterable[str]) -> str:
    return "\n\n".join([section for section in sections if section])


def format_parameter_tooltip(
    *,
    description: str = "",
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    shortcuts: Sequence[str] = (),
) -> str:
    """Format a tooltip describing a parameter control.

    This consolidates description, recommended ranges, and shortcuts so any
    widget that consumes :class:`ParameterSpec` metadata renders consistent
    guidance for keyboard and hover interactions.
    """

    sections: list[str] = []
    if description:
        sections.append(description.strip())

    if minimum is not None or maximum is not None:
        lower = "-∞" if minimum is None else str(minimum)
        upper = "∞" if maximum is None else str(maximum)
        sections.append(
            QtCore.QCoreApplication.translate(
                "ParameterSpec", "Valid range: {lower} – {upper}"
            ).format(lower=lower, upper=upper)
        )

    if shortcuts:
        shortcut_text = ", ".join(shortcuts)
        sections.append(
            QtCore.QCoreApplication.translate(
                "ParameterSpec", "Shortcuts: {shortcut_text}"
            ).format(shortcut_text=shortcut_text)
        )

    return _join_sections(sections)


def _pipeline_snapshot(controller: Optional["PipelineController"]) -> str:
    if controller is None:
        return QtCore.QCoreApplication.translate(
            "ToolTips", "No pipeline metadata available yet."
        )

    manager = controller.manager
    step_names = manager.get_order()
    if not step_names:
        return QtCore.QCoreApplication.translate(
            "ToolTips", "Current workspace has no configured steps."
        )

    preview = ", ".join(step_names[:4])
    if len(step_names) > 4:
        preview = f"{preview}, …"

    return QtCore.QCoreApplication.translate(
        "ToolTips", "Current workspace includes {count} step(s): {preview}"
    ).format(count=len(step_names), preview=preview)


def _history_snapshot(controller: Optional["PipelineController"]) -> str:
    if controller is None:
        return QtCore.QCoreApplication.translate(
            "ToolTips", "History becomes available after modifying the pipeline."
        )

    manager = controller.manager
    undo_depth, redo_depth = manager.history_depth()
    if not undo_depth and not redo_depth:
        return QtCore.QCoreApplication.translate(
            "ToolTips", "History is empty; adjust module parameters to populate it."
        )

    return QtCore.QCoreApplication.translate(
        "ToolTips", "History depth: {undo} undo / {redo} redo step(s)."
    ).format(undo=undo_depth, redo=redo_depth)


def build_main_window_tooltips(
    controller: Optional["PipelineController"],
) -> dict[str, str]:
    """Construct descriptive tooltips for :class:`~yam_processor.ui.main_window.MainWindow`.

    The strings surface pipeline-aware guidance and are shared by menus, docks,
    and any future toolbars via their underlying :class:`QtWidgets.QAction`.
    """

    return {
        "open_project": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Load a saved pipeline project, replacing the current module order and parameter presets.",
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Use when switching workflows or restoring a tuned template."
                ),
                _pipeline_snapshot(controller),
            )
        ),
        "save_project": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Write the active pipeline—including module sequence and parameter values—to its project file.",
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Trigger this before long runs or quitting to preserve adjustments.",
                ),
                _pipeline_snapshot(controller),
            )
        ),
        "save_project_as": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Persist the active pipeline to a new project file without overwriting the original.",
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Use to branch experiments or capture milestones of your tuning.",
                ),
                _pipeline_snapshot(controller),
            )
        ),
        "exit": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Close Yam Image Processor after offering to save unsaved pipeline changes."
                ),
                _pipeline_snapshot(controller),
            )
        ),
        "undo": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Restore the previous pipeline configuration from history, including module order and parameters.",
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Ideal for stepping back after testing an adjustment."
                ),
                _history_snapshot(controller),
            )
        ),
        "redo": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Reapply the next recorded pipeline state that was undone."
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Use to reintroduce a change after reviewing alternatives."
                ),
                _history_snapshot(controller),
            )
        ),
        "manage_modules": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Open the module manager to enable, disable, or reorder available processing modules.",
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Use to curate which steps appear in the pipeline palette before building workflows."
                ),
                _pipeline_snapshot(controller),
            )
        ),
        "documentation": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Open the user guide for module descriptions and workflow tutorials."
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Keep handy when learning recommended parameter ranges."
                ),
            )
        ),
        "about": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Show version and licensing details for Yam Image Processor."
                ),
            )
        ),
        "pipeline_toggle": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Show or hide the pipeline dock used for ordering modules and editing parameters."
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Keep it visible while iterating on step configuration.",
                ),
                _pipeline_snapshot(controller),
            )
        ),
        "preview_toggle": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Show or hide the preview dock displaying the most recent pipeline output."
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips",
                    "Use it to compare before/after results as you tweak module parameters.",
                ),
            )
        ),
        "diagnostics_toggle": _join_sections(
            (
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Show or hide diagnostics with logs and performance metrics."
                ),
                QtCore.QCoreApplication.translate(
                    "ToolTips", "Helpful when a module behaves unexpectedly or runs slowly."
                ),
            )
        ),
    }
