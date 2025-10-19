import json
import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtWidgets
except ImportError as exc:  # pragma: no cover - skip when Qt bindings unavailable
    QtWidgets = None  # type: ignore[assignment]
    pytestmark = pytest.mark.skip(reason=f"PyQt5 unavailable: {exc}")
else:
    from plugins.module_base import ModuleStage
    from ui.extraction import ExtractionPane
    from ui.unified import UnifiedMainWindow


class StubSettingsBackend:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def contains(self, key: str) -> bool:
        return key in self._values

    def setValue(self, key: str, value: object) -> None:
        self._values[key] = value

    def value(self, key: str, default: object | None = None) -> object | None:
        return self._values.get(key, default)


class StubSettingsManager:
    def __init__(self) -> None:
        self.backend = StubSettingsBackend()

    def snapshot(self, prefix: str | None = None) -> dict[str, object]:
        data = dict(self.backend._values)
        if prefix is None:
            return data
        return {key: value for key, value in data.items() if key.startswith(prefix)}

    def to_json(self, prefix: str | None = None, strip_prefix: bool = False) -> str:
        data = self.snapshot(prefix)
        if strip_prefix and prefix:
            prefix_len = len(prefix)
            data = {key[prefix_len:]: value for key, value in data.items()}
        return json.dumps(data)

    def from_json(
        self,
        payload: str,
        prefix: str | None = None,
        clear: bool = False,
    ) -> None:
        data = json.loads(payload)
        if clear and prefix is not None:
            for key in list(self.backend._values):
                if key.startswith(prefix):
                    del self.backend._values[key]
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix and not key.startswith(prefix) else key
            self.backend._values[full_key] = value


class StubIOManager:
    def export_preferences(self) -> dict[str, object]:
        return {}

    class _Result:
        def __init__(self, image_path: str) -> None:
            self.image_path = image_path

    def save_image(self, path, image, **kwargs):  # pragma: no cover - not exercised
        return self._Result(str(path))


class StubAppCore:
    def __init__(self) -> None:
        self._settings = StubSettingsManager()
        self._io_manager = StubIOManager()

    def ensure_bootstrapped(self) -> None:
        pass

    @property
    def settings(self) -> StubSettingsManager:
        return self._settings

    @property
    def io_manager(self) -> StubIOManager:
        return self._io_manager


if QtWidgets is not None:

    class DummyPane(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()

        def on_activated(self) -> None:
            pass

        def on_deactivated(self) -> None:
            pass

        def load_image(self) -> None:
            pass

        def save_outputs(self) -> None:
            pass

        def update_pipeline_summary(self) -> None:
            pass

        def set_diagnostics_visible(self, visible: bool) -> None:
            pass

        def teardown(self) -> None:
            pass


def _ensure_qapp() -> "QtWidgets.QApplication":
    if QtWidgets is None:  # pragma: no cover - guarded by skip markers
        raise RuntimeError("PyQt5 unavailable")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 unavailable")
def test_extraction_pane_initialises_in_tab_widget():
    _ensure_qapp()
    tab_widget = QtWidgets.QTabWidget()
    pane = ExtractionPane(StubAppCore(), parent=tab_widget)
    tab_widget.addTab(pane, "Extraction")

    assert pane.pipeline_label.text() == "Current Pipeline: (none)"

    pane.on_activated()
    pane.update_pipeline_summary()
    pane.set_diagnostics_visible(True)

    tab_widget.deleteLater()


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 unavailable")
def test_extraction_pane_lifecycle_with_unified_window():
    _ensure_qapp()
    app_core = StubAppCore()
    unified = UnifiedMainWindow(app_core)

    pane = ExtractionPane(app_core)
    pane.on_activated = MagicMock(wraps=pane.on_activated)
    pane.on_deactivated = MagicMock(wraps=pane.on_deactivated)
    pane.set_diagnostics_visible = MagicMock(wraps=pane.set_diagnostics_visible)

    unified.add_stage_pane(ModuleStage.ANALYSIS, pane, "Extraction")
    QtWidgets.QApplication.processEvents()

    assert pane.on_activated.call_count == 1
    initial_visibility_calls = pane.set_diagnostics_visible.call_count
    assert initial_visibility_calls == 1

    unified._diagnostics_dock.setVisible(False)
    QtWidgets.QApplication.processEvents()
    assert pane.set_diagnostics_visible.call_count == initial_visibility_calls + 1

    unified._diagnostics_dock.setVisible(True)
    QtWidgets.QApplication.processEvents()
    assert pane.set_diagnostics_visible.call_count == initial_visibility_calls + 2

    dummy = DummyPane()
    unified.add_stage_pane(ModuleStage.PREPROCESSING, dummy, "Dummy")
    QtWidgets.QApplication.processEvents()
    unified._tab_widget.setCurrentIndex(1)
    QtWidgets.QApplication.processEvents()

    assert pane.on_deactivated.call_count == 1

    unified.deleteLater()
