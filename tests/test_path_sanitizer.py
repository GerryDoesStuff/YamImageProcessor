"""Unit tests for the path sanitiser utilities."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core.path_sanitizer import (
    ROOT_PLACEHOLDER,
    PathValidationError,
    allowed_roots,
    configure_allowed_roots,
    redact_path_for_metadata,
    root_index_for_path,
    sanitize_user_path,
)


def test_configure_allowed_roots_creates_directories(tmp_path: Path) -> None:
    missing = tmp_path / "sandbox"
    configure_allowed_roots([missing])
    assert missing.exists()
    assert allowed_roots() == (missing.resolve(),)


def test_sanitize_user_path_allows_within_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_allowed_roots([tmp_path])
    target_dir = tmp_path / "data"
    target_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    resolved = sanitize_user_path(
        os.fspath(Path("data") / "image.png"),
        must_exist=False,
        allow_directory=False,
        allow_file=True,
    )
    assert resolved == (target_dir / "image.png").resolve()


def test_sanitize_user_path_rejects_outside_root(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path / "root"])
    outside = tmp_path / "outside" / "file.txt"
    outside.parent.mkdir()
    with pytest.raises(PathValidationError):
        sanitize_user_path(outside, must_exist=False, allow_file=True)


def test_sanitize_user_path_rejects_symlinks(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path])
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(PathValidationError):
        sanitize_user_path(link / "file.dat", must_exist=False, allow_file=True)


def test_metadata_redaction_uses_placeholder(tmp_path: Path) -> None:
    configure_allowed_roots([tmp_path])
    destination = tmp_path / "nested" / "asset.png"
    destination.parent.mkdir()
    redacted = redact_path_for_metadata(destination)
    assert redacted.startswith(ROOT_PLACEHOLDER)
    assert destination.name in redacted
    assert root_index_for_path(destination) == 0

