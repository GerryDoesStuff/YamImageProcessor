import json
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if "cv2" not in sys.modules:  # pragma: no cover - testing shim
    sys.modules["cv2"] = SimpleNamespace(
        imwrite=lambda *args, **kwargs: True,
        IMREAD_UNCHANGED=1,
        imread=lambda *args, **kwargs: None,
    )

np = pytest.importorskip("numpy")

from core.io_manager import SaveResult
from core.persistence import AutosaveManager
from core.recovery import RecoveryManager


class DummySettings:
    def __init__(self, *, interval: float = 0.0, enabled: bool = True, retention: int = 2) -> None:
        self._interval = interval
        self._enabled = enabled
        self._retention = retention

    def autosave_enabled(self) -> bool:
        return self._enabled

    def autosave_interval(self) -> float:
        return self._interval

    def autosave_backup_retention(self) -> int:
        return self._retention

    def snapshot(self) -> dict[str, object]:
        return {"settings": "snapshot"}


class DummyIOManager:
    default_format = ".dat"

    def __init__(self) -> None:
        self._backup_counter = 0

    def save_image(
        self,
        destination: Path,
        image: np.ndarray,
        *,
        metadata: dict,
        pipeline: dict,
        settings_snapshot: dict,
        create_backup: bool,
        backup_retention: int,
    ) -> SaveResult:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous_bytes = destination.read_bytes() if destination.exists() else None
        destination.write_bytes(image.tobytes())
        sidecar = {
            "schema": "test.autosave",
            "image": {
                "filename": destination.name,
                "path": str(destination),
                "display_path": str(destination),
                "root_index": 0,
                "format": "DUMMY",
            },
            "metadata": dict(metadata),
            "pipeline": dict(pipeline),
            "settings": dict(settings_snapshot),
        }
        metadata_path = destination.with_suffix(".json")
        metadata_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
        if create_backup and previous_bytes is not None:
            backup_dir = destination.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            self._backup_counter += 1
            backup_path = backup_dir / f"autosave-{self._backup_counter}.bak"
            backup_path.write_bytes(previous_bytes)
            if backup_retention >= 0:
                backups = sorted(backup_dir.glob("*.bak"), key=lambda item: item.stat().st_mtime)
                while len(backups) > backup_retention:
                    stale = backups.pop(0)
                    stale.unlink(missing_ok=True)
                    backups = sorted(backup_dir.glob("*.bak"), key=lambda item: item.stat().st_mtime)
        return SaveResult(destination, metadata_path)


class FailingIOManager(DummyIOManager):
    def save_image(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("simulated failure")


def _create_manager(
    tmp_path: Path, *, io_manager
) -> tuple[AutosaveManager, RecoveryManager, Path, DummySettings]:
    autosave_dir = tmp_path / "autosave"
    recovery_root = tmp_path / "recovery"
    recovery = RecoveryManager(autosave_dir, recovery_root=recovery_root)
    recovery.inspect_startup()
    settings = DummySettings()
    manager = AutosaveManager(
        settings,
        io_manager,
        autosave_directory=autosave_dir,
        interval_seconds=0,
        logger=logging.getLogger("test.autosave"),
        recovery_manager=recovery,
    )
    return manager, recovery, recovery_root, settings


def test_autosave_success_clears_crash_marker_and_updates_summary(tmp_path):
    manager, recovery, recovery_root, settings = _create_manager(
        tmp_path, io_manager=DummyIOManager()
    )
    image = np.zeros((2, 2), dtype=np.uint8)
    pipeline = {"stage": "unit"}
    manager.mark_dirty(image, pipeline, metadata={"note": "initial"})

    crash_marker_dir = recovery_root / "crash_markers"
    assert not list(crash_marker_dir.glob("pending_autosave_*.json"))

    summary = recovery.summary()
    assert summary.has_snapshot is True
    assert summary.snapshot is not None
    assert summary.snapshot.image_path == manager.last_result().image_path

    for _ in range(4):
        manager.mark_dirty(image, pipeline)

    retention = settings.autosave_backup_retention()
    backups = list((manager.last_result().image_path.parent / "backups").glob("*.bak"))
    assert len(backups) <= retention


def test_autosave_failure_retains_crash_marker(tmp_path):
    manager, recovery, recovery_root, _settings = _create_manager(
        tmp_path, io_manager=FailingIOManager()
    )
    image = np.zeros((2, 2), dtype=np.uint8)
    pipeline = {"stage": "unit"}
    manager.mark_dirty(image, pipeline)

    crash_marker_dir = recovery_root / "crash_markers"
    markers = list(crash_marker_dir.glob("pending_autosave_*.json"))
    assert markers, "Crash marker should remain when autosave fails"


def test_project_save_uses_guarded_marker(tmp_path):
    io_manager = DummyIOManager()
    manager, recovery, recovery_root, _settings = _create_manager(
        tmp_path, io_manager=io_manager
    )
    image = np.zeros((2, 2), dtype=np.uint8)
    pipeline = {"stage": "unit"}
    manager.mark_dirty(image, pipeline)

    destination = tmp_path / "project" / "result.dat"
    result = manager.save(destination, image=image, pipeline=pipeline, metadata={"note": "save"})
    assert result.image_path == destination

    crash_marker_dir = recovery_root / "crash_markers"
    assert not list(crash_marker_dir.glob("pending_project_save_*.json"))
