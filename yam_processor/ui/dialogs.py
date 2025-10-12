"""Reusable Qt dialogs and widgets for parameterised pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore


class ParameterType(Enum):
    """Supported control widget types for :class:`ParameterDialog`."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    CHOICE = "choice"


@dataclass(frozen=True)
class ParameterSpec:
    """Declarative description of a parameter control."""

    name: str
    label: str
    type: ParameterType
    default: Any = None
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[Sequence[tuple[str, Any]]] = None
    shortcuts: Sequence[str] = ()
    placeholder: Optional[str] = None
    decimals: int = 2

    def tooltip(self) -> str:
        """Generate a descriptive tooltip including range and shortcuts."""

        sections: list[str] = []
        if self.description:
            sections.append(self.description.strip())

        if self.minimum is not None or self.maximum is not None:
            lower = "-∞" if self.minimum is None else str(self.minimum)
            upper = "∞" if self.maximum is None else str(self.maximum)
            sections.append(f"Valid range: {lower} – {upper}")

        if self.shortcuts:
            shortcut_text = ", ".join(self.shortcuts)
            sections.append(f"Shortcuts: {shortcut_text}")

        return "\n\n".join(sections)


def _apply_parameter_metadata(widget: QtWidgets.QWidget, spec: ParameterSpec) -> None:
    tooltip = spec.tooltip()
    if tooltip:
        widget.setToolTip(tooltip)
        widget.setWhatsThis(tooltip)


class PreviewWidget(QtWidgets.QWidget):
    """Display widget that renders numpy arrays into a ``QGraphicsView``."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = QtWidgets.QGraphicsView(self._scene, self)
        self._view.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self._view.setAlignment(QtCore.Qt.AlignCenter)
        self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._current_pixmap: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._image_buffer: Optional[np.ndarray] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        """Render ``image`` or clear the view if ``None`` is supplied."""

        self._scene.clear()
        self._current_pixmap = None
        self._image_buffer = None
        if image is None:
            return

        qimage, buffer = self._to_qimage(image)
        self._image_buffer = buffer
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self._current_pixmap = self._scene.addPixmap(pixmap)
        self._fit_view()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_view()

    def _fit_view(self) -> None:
        if self._current_pixmap is None:
            return
        rect = self._current_pixmap.sceneBoundingRect()
        if rect.isNull():
            return
        self._view.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    @staticmethod
    def _to_qimage(image: np.ndarray) -> tuple[QtGui.QImage, np.ndarray]:
        if image.ndim not in (2, 3):
            raise ValueError("PreviewWidget expects 2D or 3D numpy arrays")

        array = np.ascontiguousarray(image)

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            height, width = array.shape
            qimage = QtGui.QImage(array.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            return qimage, array

        height, width, channels = array.shape
        if channels == 3:
            qimage = QtGui.QImage(array.data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
            return qimage.rgbSwapped(), array
        if channels == 4:
            qimage = QtGui.QImage(
                array.data, width, height, 4 * width, QtGui.QImage.Format_RGBA8888
            )
            return qimage, array
        raise ValueError("Unsupported channel count for preview rendering")


class _PreviewWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(Exception)


class _PreviewRunnable(QtCore.QRunnable):
    def __init__(
        self,
        preview_callable: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.preview_callable = preview_callable
        self.image = image
        self.params = params
        self.signals = _PreviewWorkerSignals()

    def run(self) -> None:  # noqa: D401 - Qt interface
        try:
            result = self.preview_callable(self.image, self.params)
        except Exception as exc:  # pragma: no cover - Qt threading
            self.signals.failed.emit(exc)
            return
        self.signals.finished.emit(result)


class ParameterDialog(QtWidgets.QDialog):
    """Modeless dialog that previews module parameter changes."""

    cancelled = QtCore.pyqtSignal()
    parametersChanged = QtCore.pyqtSignal(dict)
    previewUpdated = QtCore.pyqtSignal(np.ndarray)
    previewFailed = QtCore.pyqtSignal(Exception)

    def __init__(
        self,
        schema: Iterable[ParameterSpec],
        preview_callback: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        window_title: str = "Module Parameters",
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(window_title)

        self._schema: list[ParameterSpec] = list(schema)
        self._preview_callback = preview_callback
        self._source_image: Optional[np.ndarray] = None
        self._controls: dict[str, QtWidgets.QWidget] = {}
        self._updating_controls = False
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._preview_running = False
        self._pending_params: Optional[Dict[str, Any]] = None
        self._initial_parameters: Dict[str, Any] = {}

        self._build_ui()
        self.reset_to_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_source_image(self, image: Optional[np.ndarray]) -> None:
        self._source_image = None if image is None else np.array(image, copy=True)
        if image is None:
            self._preview_widget.set_image(None)
            return
        self._schedule_preview()

    def set_parameters(self, parameters: Mapping[str, Any]) -> None:
        self._initial_parameters = dict(parameters)
        self._apply_parameters(self._initial_parameters)

    def parameters(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for spec in self._schema:
            widget = self._controls.get(spec.name)
            if widget is None:
                continue
            values[spec.name] = self._get_widget_value(spec, widget)
        return values

    def reset_to_defaults(self) -> None:
        defaults = {spec.name: spec.default for spec in self._schema}
        self._apply_parameters(defaults)

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------
    def reject(self) -> None:  # noqa: D401 - Qt override
        self.cancelled.emit()
        super().reject()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        control_container = QtWidgets.QWidget(splitter)
        control_layout = QtWidgets.QFormLayout(control_container)
        control_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        control_layout.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        control_layout.setSpacing(8)

        for spec in self._schema:
            widget = self._create_widget_for_spec(spec)
            self._controls[spec.name] = widget
            control_layout.addRow(spec.label, widget)

        splitter.addWidget(control_container)

        preview_container = QtWidgets.QWidget(splitter)
        preview_layout = QtWidgets.QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        self._preview_widget = PreviewWidget(preview_container)
        preview_layout.addWidget(self._preview_widget, 1)
        splitter.addWidget(preview_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        self._status_label = QtWidgets.QLabel("Adjust parameters to preview changes", self)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel, self)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_widget_for_spec(self, spec: ParameterSpec) -> QtWidgets.QWidget:
        if spec.type == ParameterType.INTEGER:
            widget = QtWidgets.QSpinBox(self)
            if spec.minimum is not None:
                widget.setMinimum(int(spec.minimum))
            if spec.maximum is not None:
                widget.setMaximum(int(spec.maximum))
            if spec.step is not None:
                widget.setSingleStep(int(spec.step))
            widget.valueChanged.connect(self._on_parameter_changed)
        elif spec.type == ParameterType.FLOAT:
            widget = QtWidgets.QDoubleSpinBox(self)
            widget.setDecimals(spec.decimals)
            if spec.minimum is not None:
                widget.setMinimum(float(spec.minimum))
            if spec.maximum is not None:
                widget.setMaximum(float(spec.maximum))
            if spec.step is not None:
                widget.setSingleStep(float(spec.step))
            widget.valueChanged.connect(self._on_parameter_changed)
        elif spec.type == ParameterType.BOOLEAN:
            checkbox = QtWidgets.QCheckBox(self)
            checkbox.toggled.connect(self._on_parameter_changed)
            widget = checkbox
        elif spec.type == ParameterType.CHOICE:
            combo = QtWidgets.QComboBox(self)
            combo.setEditable(False)
            for label, value in spec.choices or []:
                combo.addItem(label, value)
            combo.currentIndexChanged.connect(self._on_parameter_changed)
            widget = combo
        else:
            line_edit = QtWidgets.QLineEdit(self)
            if spec.placeholder:
                line_edit.setPlaceholderText(spec.placeholder)
            line_edit.textChanged.connect(self._on_parameter_changed)
            widget = line_edit

        _apply_parameter_metadata(widget, spec)
        return widget

    def _apply_parameters(self, parameters: Mapping[str, Any]) -> None:
        self._updating_controls = True
        try:
            for spec in self._schema:
                value = parameters.get(spec.name, spec.default)
                widget = self._controls.get(spec.name)
                if widget is None:
                    continue
                self._set_widget_value(spec, widget, value)
        finally:
            self._updating_controls = False
        self._on_parameter_changed()

    def _set_widget_value(self, spec: ParameterSpec, widget: QtWidgets.QWidget, value: Any) -> None:
        if value is None:
            value = spec.default
        if isinstance(widget, QtWidgets.QSpinBox):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.blockSignals(True)
            widget.setChecked(bool(value))
            widget.blockSignals(False)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.blockSignals(True)
            index = widget.findData(value)
            if index < 0 and isinstance(value, str):
                index = widget.findText(value)
            if index >= 0:
                widget.setCurrentIndex(index)
            widget.blockSignals(False)
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.blockSignals(True)
            widget.setText("" if value is None else str(value))
            widget.blockSignals(False)

    def _get_widget_value(self, spec: ParameterSpec, widget: QtWidgets.QWidget) -> Any:
        if isinstance(widget, QtWidgets.QSpinBox):
            return int(widget.value())
        if isinstance(widget, QtWidgets.QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QtWidgets.QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QtWidgets.QComboBox):
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        raise TypeError(f"Unsupported widget type for parameter '{spec.name}'")

    @QtCore.pyqtSlot()
    def _on_parameter_changed(self) -> None:
        if self._updating_controls:
            return
        params = self.parameters()
        self.parametersChanged.emit(params)
        self._pending_params = params
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._preview_callback is None or self._source_image is None:
            return
        if self._preview_running:
            return
        if self._pending_params is None:
            self._pending_params = self.parameters()
        self._start_preview(self._pending_params)
        self._pending_params = None

    def _start_preview(self, params: Dict[str, Any]) -> None:
        if self._preview_callback is None or self._source_image is None:
            return
        self._status_label.setText("Rendering preview…")
        image = np.array(self._source_image, copy=True)
        runnable = _PreviewRunnable(self._preview_callback, image, dict(params))
        runnable.signals.finished.connect(self._handle_preview_finished)
        runnable.signals.failed.connect(self._handle_preview_failed)
        self._preview_running = True
        self._thread_pool.start(runnable)

    @QtCore.pyqtSlot(object)
    def _handle_preview_finished(self, result: object) -> None:
        self._preview_running = False
        if isinstance(result, np.ndarray):
            self._preview_widget.set_image(result)
            self.previewUpdated.emit(result)
            self._status_label.setText("Preview updated")
        else:
            self._preview_widget.set_image(None)
            self._status_label.setText("Preview unavailable")
        if self._pending_params is not None:
            params = self._pending_params
            self._pending_params = None
            self._start_preview(params)

    @QtCore.pyqtSlot(Exception)
    def _handle_preview_failed(self, error: Exception) -> None:  # pragma: no cover - Qt threading
        self._preview_running = False
        self.previewFailed.emit(error)
        self._status_label.setText(f"Preview error: {error}")
        if self._pending_params is not None:
            params = self._pending_params
            self._pending_params = None
            self._start_preview(params)


__all__ = [
    "ParameterDialog",
    "ParameterSpec",
    "ParameterType",
    "PreviewWidget",
]

