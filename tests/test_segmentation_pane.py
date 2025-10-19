from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtWidgets
except ImportError as exc:  # pragma: no cover - skip if Qt bindings missing
    QtWidgets = None  # type: ignore[assignment]
    pytestmark = pytest.mark.skip(reason=f"PyQt5 unavailable: {exc}")
else:
    from ui.segmentation import SegmentationPane


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


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_segmentation_pane_initialises_in_tab_widget():
    _ensure_qapp()
    tab_widget = QtWidgets.QTabWidget()
    pane = SegmentationPane(StubAppCore(), parent=tab_widget)
    tab_widget.addTab(pane, "Segmentation")

    assert pane.pipeline_label.text() == "Current Pipeline: (none)"
    assert pane.image_splitter.count() == 2

    pane.on_activated()
    pane.update_pipeline_summary()
    pane.set_diagnostics_visible(True)

    tab_widget.deleteLater()
