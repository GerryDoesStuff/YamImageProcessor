from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.settings import SettingsManager


def test_settings_json_roundtrip(tmp_path: Path) -> None:
    manager = SettingsManager("TestOrg", "TestApp", seed_defaults=False)
    manager.set("alpha", 123)
    manager.set("beta", {"nested": [1, 2, 3]})

    export_path = tmp_path / "settings.json"
    manager.export_json(export_path)
    assert export_path.exists()

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported == {"alpha": 123, "beta": {"nested": [1, 2, 3]}}

    manager.clear()
    assert manager.get("alpha") is None
    assert manager.get("beta") is None

    manager.import_json(export_path)
    assert manager.get("alpha") == 123
    assert manager.get("beta") == {"nested": [1, 2, 3]}


def test_settings_from_dict_with_clear() -> None:
    manager = SettingsManager("AnotherOrg", "AnotherApp", seed_defaults=False)
    manager.set("keep", "value")

    manager.from_dict({"fresh": 42}, clear=True)
    assert manager.get("keep") is None
    assert manager.get("fresh") == 42
    assert manager.to_dict() == {"fresh": 42}
