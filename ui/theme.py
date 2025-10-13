"""Shared theming helpers for YamImageProcessor Qt applications."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


_SECTION_MARGINS = (12, 12, 12, 12)
_SECTION_SPACING = 8


@lru_cache(maxsize=1)
def _icon_search_paths() -> Tuple[Path, ...]:
    """Return directories that may contain application icons."""

    base_dir = Path(__file__).resolve().parent
    candidates: List[Path] = [base_dir / "icons"]
    candidates.append(base_dir.parent / "yam_processor" / "ui" / "icons")
    return tuple(path for path in candidates if path.exists())


@lru_cache(maxsize=None)
def load_icon(
    name: str,
    *,
    fallback: QtGui.QIcon | None = None,
    size: int = 24,
) -> QtGui.QIcon:
    """Load an application icon and return a scalable :class:`~QtGui.QIcon`."""

    suffixes = (".svg", ".png", ".ico")
    icon = QtGui.QIcon()

    for directory in _icon_search_paths():
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if not candidate.exists():
                continue
            if candidate.suffix.lower() == ".svg":
                icon.addFile(str(candidate), QtCore.QSize(size, size))
            else:
                pixmap = QtGui.QPixmap(str(candidate))
                if not pixmap.isNull() and size > 0:
                    pixmap = pixmap.scaled(
                        size,
                        size,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                icon.addPixmap(pixmap)
        if not icon.isNull():
            break

    if icon.isNull():
        icon = fallback or QtGui.QIcon()
    return icon


def create_high_contrast_palette() -> QtGui.QPalette:
    """Build a high-contrast palette suitable for dark-themed workflows."""

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#121212"))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1e1e"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#252525"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#111111"))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#202020"))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor("#58a6ff"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#0a84ff"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    return palette


def apply_application_theme(app: QtWidgets.QApplication) -> None:
    """Apply the shared theme to the given :class:`QApplication`."""

    app.setStyle("Fusion")
    app.setPalette(create_high_contrast_palette())
    font = app.font()
    if font.pointSize() > 0:
        font.setPointSize(max(font.pointSize() + 1, int(font.pointSize() * 1.1)))
    app.setFont(font)


def scale_font(
    base: QtGui.QFont,
    *,
    factor: float = 1.0,
    weight: int | None = None,
) -> QtGui.QFont:
    """Return a copy of *base* scaled by *factor* with an optional *weight*."""

    clone = QtGui.QFont(base)
    if clone.pointSizeF() > 0:
        clone.setPointSizeF(clone.pointSizeF() * factor)
    elif clone.pointSize() > 0:
        clone.setPointSize(max(int(clone.pointSize() * factor), clone.pointSize() + 1))
    if weight is not None:
        clone.setWeight(weight)
    return clone


class SectionWidget(QtWidgets.QWidget):
    """Container widget with a themed heading and consistent spacing."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(*_SECTION_MARGINS)
        self._layout.setSpacing(_SECTION_SPACING)
        self.header_label = QtWidgets.QLabel(title, self)
        self.header_label.setObjectName("sectionHeader")
        base_font = QtWidgets.QApplication.font()
        self.header_label.setFont(
            scale_font(base_font, factor=1.2, weight=QtGui.QFont.DemiBold)
        )
        self.header_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.header_label.setAccessibleDescription(f"Section heading: {title}")
        self._layout.addWidget(self.header_label)

    @property
    def layout(self) -> QtWidgets.QVBoxLayout:
        return self._layout


class ThemedDockWidget(QtWidgets.QDockWidget):
    """Dock widget that respects the global theme and accessibility defaults."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)
        self.setObjectName(f"ThemedDock_{title.replace(' ', '_')}")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )


class ShortcutSummaryWidget(QtWidgets.QTreeWidget):
    """Compact tree widget that lists registered keyboard shortcuts."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Action", "Shortcut"])
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setObjectName("shortcutSummary")
        self.setAccessibleName("Keyboard shortcut summary")
        self.setUniformRowHeights(True)
        self.header().setStretchLastSection(True)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

    def populate(self, entries: Sequence[Tuple[str, str]]) -> None:
        self.clear()
        for description, shortcut in entries:
            item = QtWidgets.QTreeWidgetItem([description, shortcut])
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.addTopLevelItem(item)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)


class ShortcutRegistry(QtCore.QObject):
    """Track shortcut assignments and publish them to the UI."""

    def __init__(
        self,
        summary_widget: ShortcutSummaryWidget | None = None,
        status_label: QtWidgets.QLabel | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._entries: List[Tuple[str, str]] = []
        self._summary_widget = summary_widget
        self._status_label = status_label

    def register_action(self, description: str, action: QtWidgets.QAction) -> None:
        sequences = action.shortcuts() or [action.shortcut()]
        self._register_sequences(description, sequences)

    def register_shortcut(self, description: str, shortcut: QtWidgets.QShortcut) -> None:
        sequences = [shortcut.key()]
        self._register_sequences(description, sequences)

    def reset(self) -> None:
        """Clear previously registered shortcuts."""

        self._entries.clear()
        self._publish()

    def _register_sequences(self, description: str, sequences: Iterable[QtGui.QKeySequence]) -> None:
        for sequence in sequences:
            if not sequence:
                continue
            text = sequence.toString(QtGui.QKeySequence.NativeText)
            if not text:
                continue
            self._entries.append((description, text))
        self._publish()

    def _publish(self) -> None:
        unique_entries: List[Tuple[str, str]] = []
        seen = set()
        for description, shortcut in self._entries:
            key = (description, shortcut)
            if key in seen:
                continue
            seen.add(key)
            unique_entries.append((description, shortcut))

        if self._summary_widget is not None:
            self._summary_widget.populate(unique_entries)
        if self._status_label is not None:
            summary = "  |  ".join(f"{desc}: {shortcut}" for desc, shortcut in unique_entries[:6])
            self._status_label.setText(summary)


__all__ = [
    "SectionWidget",
    "ThemedDockWidget",
    "ShortcutRegistry",
    "ShortcutSummaryWidget",
    "apply_application_theme",
    "create_high_contrast_palette",
    "load_icon",
    "scale_font",
]
