import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "cv2" not in sys.modules:  # pragma: no cover - testing shim
    sys.modules["cv2"] = SimpleNamespace(
        imwrite=lambda *args, **kwargs: True,
        IMREAD_UNCHANGED=1,
        imread=lambda *args, **kwargs: None,
    )

from core.recovery import RecoveryManager


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_autosave_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    autosave_dir = tmp_path / "autosave"
    autosave_dir.mkdir()
    image_path = autosave_dir / "autosave.png"
    image_path.write_bytes(b"fake-image-bytes")
    metadata_path = autosave_dir / "autosave.json"
    _write_json(
        metadata_path,
        {
            "image": {
                "filename": image_path.name,
                "path": str(image_path),
                "format": "PNG",
            },
            "metadata": {"destination": "project.yam"},
            "pipeline": {"steps": []},
            "settings": {},
        },
    )
    backups_dir = autosave_dir / "backups"
    backups_dir.mkdir()
    backup_path = backups_dir / "autosave-1.png"
    backup_path.write_bytes(b"backup")
    return autosave_dir, image_path, backup_path


def test_inspect_startup_discovers_snapshot_and_crash_markers(tmp_path):
    autosave_dir, image_path, backup_path = _prepare_autosave_workspace(tmp_path)
    recovery_root = tmp_path / "recovery"
    crash_marker_dir = recovery_root / "crash_markers"
    crash_marker_dir.mkdir(parents=True)
    stale_marker = crash_marker_dir / "previous.json"
    _write_json(stale_marker, {"reason": "forced_shutdown"})

    manager = RecoveryManager(autosave_dir, recovery_root=recovery_root)
    snapshot = manager.inspect_startup()

    assert snapshot is not None
    assert snapshot.image_path == image_path
    assert snapshot.metadata_path == autosave_dir / "autosave.json"
    assert snapshot.metadata["metadata"]["destination"] == "project.yam"
    assert snapshot.backups == (backup_path,)
    assert manager.has_pending_snapshot() is True
    assert manager.crash_detected() is True
    markers = manager.crash_markers()
    assert len(markers) == 1
    assert markers[0].path.name == "previous.json"
    assert (recovery_root / "crash_markers" / "previous.json").exists() is False
    session_markers = list((recovery_root / "crash_markers").glob("session_*.json"))
    assert session_markers, "session marker should be written for current run"


def test_discard_pending_removes_autosave_files(tmp_path):
    autosave_dir, image_path, backup_path = _prepare_autosave_workspace(tmp_path)
    recovery_root = tmp_path / "recovery"
    manager = RecoveryManager(autosave_dir, recovery_root=recovery_root)
    manager.inspect_startup()

    snapshot = manager.discard_pending()
    assert snapshot is not None
    assert manager.has_pending_snapshot() is False
    assert image_path.exists() is False
    assert (autosave_dir / "autosave.json").exists() is False
    assert backup_path.exists() is False
    assert (autosave_dir / "backups").exists() is False


def test_cleanup_crash_markers_removes_session_marker(tmp_path):
    autosave_dir = tmp_path / "autosave"
    autosave_dir.mkdir()
    recovery_root = tmp_path / "recovery"
    manager = RecoveryManager(autosave_dir, recovery_root=recovery_root)
    manager.inspect_startup()

    marker_dir = recovery_root / "crash_markers"
    session_markers = list(marker_dir.glob("session_*.json"))
    assert session_markers

    manager.cleanup_crash_markers()

    for marker in session_markers:
        assert marker.exists() is False
    assert marker_dir.exists() is False


def test_summary_reports_pending_state(tmp_path):
    autosave_dir, image_path, _backup_path = _prepare_autosave_workspace(tmp_path)
    manager = RecoveryManager(autosave_dir)
    manager.inspect_startup()

    summary = manager.summary()
    assert summary.has_snapshot is True
    assert summary.snapshot is not None
    assert summary.snapshot.image_path == image_path

    metadata = summary.to_metadata()
    assert metadata["has_snapshot"] is True
    assert metadata["crash_marker_count"] == 0

    status = summary.status_message()
    assert status is not None
    message, is_error = status
    assert str(autosave_dir) in message
    assert is_error is False
